"""
trajectory_planner.py
------------------
Real safe-path / trajectory guidance, replacing the plain circle+arrow
direction indicator with an actual perspective road path.

Works together with direction_predictor.py:
    - direction_predictor.py decides WHICH way (STRAIGHT / LEFT / RIGHT)
      is safe, using obstacle positions + free-corridor analysis.
    - trajectory_planner.py turns that decision into an actual curved
      path on the road: a quadratic-Bezier curve from the bottom-center
      of the camera view toward the chosen corridor, clipped to the
      drivable-area mask and nudged away from obstacle boxes it would
      otherwise cross.

Usage:
    from trajectory_planner import TrajectoryPlanner
    planner = TrajectoryPlanner()
    ...
    pts = planner.compute_safe_trajectory(direction, drivable_mask, frame.shape, obstacle_boxes)
    frame = planner.draw_safe_trajectory(frame, pts, risk=risk_now)
"""

import cv2
import numpy as np

_RISK_COLORS = {
    "SAFE":    (0, 245, 80),
    "WARNING": (0, 180, 255),
    "DANGER":  (0, 0, 255),
}

_LATERAL_OFFSET = {"STRAIGHT": 0.0, "LEFT": -0.30, "RIGHT": 0.30}


class TrajectoryPlanner:
    """Computes + draws a smooth, obstacle-aware driving path curve.

    Temporal smoothing: the Bezier control point's lateral (x) position
    is EMA-smoothed frame to frame, exactly like direction_predictor's
    soft-vote, so the path eases toward a new direction instead of
    snapping/flickering.
    """

    def __init__(self, alpha: float = 0.20, look_ahead_frac: float = 0.42):
        self._alpha = alpha
        self._smoothed_cx = None      # smoothed target lateral pixel (control pt)
        self._look_ahead_frac = look_ahead_frac  # how far "up" the curve reaches (0-1 of frame height)

    # ------------------------------------------------------------------
    def _row_free_span(self, drivable_mask, row, frame_w):
        """Returns (min_x, max_x) of drivable pixels in a mask row, or
        None if that row has no drivable pixels (falls back to full width)."""
        if drivable_mask is None:
            return 0, frame_w - 1
        cols = np.where(drivable_mask[row])[0]
        if len(cols) == 0:
            return None
        return int(cols.min()), int(cols.max())

    def _obstacle_free_x(self, x, row_band, obstacle_boxes):
        """True if column `x` is clear of every obstacle box across the
        given row band (y0, y1)."""
        y0, y1 = row_band
        for (ox1, oy1, ox2, oy2) in obstacle_boxes:
            if oy2 < y0 or oy1 > y1:
                continue  # obstacle not in this vertical band
            if ox1 - 15 <= x <= ox2 + 15:
                return False
        return True

    # ------------------------------------------------------------------
    def compute_safe_trajectory(self, direction, drivable_mask, frame_shape,
                                 obstacle_boxes=None, n_points: int = 24):
        """
        Returns a list of (x, y) pixel points describing a smooth curved
        path from the bottom-center of the frame toward the chosen
        corridor, clipped to stay inside the drivable area and nudged
        sideways away from obstacle boxes it would otherwise cross.
        """
        obstacle_boxes = obstacle_boxes or []
        h, w = frame_shape[:2]

        start = (w / 2.0, float(h - 10))
        look_ahead_y = h * (1.0 - self._look_ahead_frac)

        # ---- pick a target lateral position within the drivable span ----
        span = self._row_free_span(drivable_mask, int(look_ahead_y), w) if drivable_mask is not None else None
        if span is not None:
            lo, hi = span
        else:
            lo, hi = int(w * 0.15), int(w * 0.85)
        span_w = max(1, hi - lo)
        span_cx = (lo + hi) / 2.0

        target_x = span_cx + _LATERAL_OFFSET.get(direction, 0.0) * span_w
        target_x = float(np.clip(target_x, lo + span_w * 0.08, hi - span_w * 0.08))

        # ---- nudge target away from any obstacle sitting right at the look-ahead band ----
        band = (look_ahead_y - h * 0.12, look_ahead_y + h * 0.12)
        tries = 0
        step = max(6.0, span_w * 0.04)
        while not self._obstacle_free_x(target_x, band, obstacle_boxes) and tries < 12:
            # walk toward whichever side of the span has more room
            if target_x < span_cx:
                target_x -= step
            else:
                target_x += step
            target_x = float(np.clip(target_x, lo + 4, hi - 4))
            tries += 1

        # ---- temporal smoothing on the control point ----
        if self._smoothed_cx is None:
            self._smoothed_cx = target_x
        else:
            self._smoothed_cx = self._alpha * target_x + (1.0 - self._alpha) * self._smoothed_cx
        ctrl_x = self._smoothed_cx
        ctrl_y = (start[1] + look_ahead_y) / 2.0

        end = (ctrl_x, look_ahead_y)
        control = (start[0] + (ctrl_x - start[0]) * 0.65, ctrl_y)

        # ---- sample quadratic Bezier: start -> control -> end ----
        pts = []
        for i in range(n_points + 1):
            t = i / n_points
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
            pts.append((x, y))
        return pts

    # ------------------------------------------------------------------
    def draw_safe_trajectory(self, frame, points, risk: str = "SAFE", label: str = None):
        """Draws the path as a tapering, glowing 'carpet' (wide near the
        car, narrowing with perspective) plus a bright centerline -- reads
        clearly as a recommended driving path, not just a thin line."""
        if not points or len(points) < 2:
            return frame

        color = _RISK_COLORS.get(risk, _RISK_COLORS["SAFE"])
        h, w = frame.shape[:2]
        n = len(points)

        overlay = frame.copy()
        # tapering polygon "carpet": wide at the bottom, narrow toward the vanishing point
        left_edge, right_edge = [], []
        for i, (x, y) in enumerate(points):
            t = i / (n - 1)
            half_w = max(4.0, 46.0 * (1.0 - t) ** 1.4 + 4.0)
            left_edge.append((x - half_w, y))
            right_edge.append((x + half_w, y))
        poly = np.array(left_edge + right_edge[::-1], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        # bright dashed centerline on top
        for i in range(0, n - 1, 2):
            p1 = tuple(np.int32(points[i]))
            p2 = tuple(np.int32(points[min(i + 1, n - 1)]))
            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)

        # arrowhead at the far end showing recommended heading
        tip = points[-1]
        prev = points[-2]
        ang = np.arctan2(tip[1] - prev[1], tip[0] - prev[0])
        tip_i = np.int32(tip)
        for da in (-2.6, 2.6):
            a = ang + da
            p = (int(tip[0] - 18 * np.cos(a)), int(tip[1] - 18 * np.sin(a)))
            cv2.line(frame, tuple(tip_i), p, color, 3, cv2.LINE_AA)

        if label:
            cv2.putText(frame, label, (int(points[-1][0]) - 40, max(20, int(points[-1][1]) - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return frame

    def reset(self):
        self._smoothed_cx = None
