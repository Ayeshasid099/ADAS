"""
direction_predictor.py
------------------
v3 — real corridor-safety direction guidance (upgrade from the old
single-vehicle / distance-only heuristic).

Decision logic (per spec):
  - No dangerous obstacle anywhere  -> STRAIGHT
  - Obstacle centered               -> pick the SAFER of the two corridors
  - Obstacle on left                -> prefer RIGHT, but ONLY if the right
                                        corridor is actually free
  - Obstacle on right               -> prefer LEFT, but ONLY if the left
                                        corridor is actually free
  - A corridor blocked by ANY obstacle (not just the primary threat) is
    NEVER recommended, even as a fallback.
  - Both corridors unsafe           -> STRAIGHT (brake handles the rest;
                                        we never invent a fake-safe direction)

"Obstacles" here is the unified list built by adas_core.py / adas_full_
pipeline.py: vehicles AND animals both use the same dict shape (box,
distance, ttc, risk, name), so this module doesn't care which kind of
object it is -- exactly what lets animals plug into the existing
direction/trajectory system with no separate code path.

Soft-vote EMA (unchanged from v2) keeps the final direction from
flickering frame to frame even as corridor safety flips near a boundary.

Usage:
    from direction_predictor import DirectionPredictor
    dp = DirectionPredictor()
    ...
    direction = dp.predict(obstacles, frame.shape, drivable_mask)
    frame = dp.draw_arrow(frame, direction, drivable_mask, risk=risk_now)
    left_ok, right_ok = dp.last_corridor_safety   # for trajectory_planner / HUD
"""

import cv2
import numpy as np


_DIR_COLORS = {
    "STRAIGHT": (0, 245, 80),
    "LEFT":     (0, 210, 255),
    "RIGHT":    (30, 160, 255),
}

_DIR_LABEL = {
    "STRAIGHT": "STRAIGHT",
    "LEFT":     "STEER LEFT",
    "RIGHT":    "STEER RIGHT",
}


class DirectionPredictor:
    def __init__(self, alpha: float = 0.22, threat_distance: float = 12.0,
                 threat_ttc: float = 4.0):
        """alpha: EMA weight for soft-vote smoothing (0.15-0.30 works well)."""
        self._votes: dict[str, float] = {"STRAIGHT": 1.0, "LEFT": 0.0, "RIGHT": 0.0}
        self._alpha = alpha
        self.current = "STRAIGHT"
        self.threat_distance = threat_distance
        self.threat_ttc = threat_ttc
        # populated each predict() call -> (left_safe: bool, right_safe: bool)
        self.last_corridor_safety = (True, True)
        self.both_unsafe = False

    # ------------------------------------------------------------------
    def _lateral_zone(self, box, frame_w: int) -> str:
        """Horizontal third of frame where the detected box's centre lies."""
        cx_norm = (box[0] + box[2]) / 2.0 / max(frame_w, 1)
        if cx_norm < 0.37:
            return "left"
        if cx_norm > 0.63:
            return "right"
        return "center"

    def _is_threat(self, obs) -> bool:
        risk = obs.get("risk", "SAFE")
        dist = obs.get("distance", 999.0)
        ttc = obs.get("ttc", float("inf"))
        closing_fast = (ttc != float("inf")) and (ttc < self.threat_ttc)
        return (risk != "SAFE") or (dist <= self.threat_distance) or closing_fast

    # ------------------------------------------------------------------
    def _corridor_free(self, side: str, drivable_mask, frame_w: int, frame_h: int,
                        obstacles) -> bool:
        """A corridor (left/right half of the drivable area, in the near-
        forward band of the frame) counts as free if:
          1. the drivable mask actually has road pixels there (if a mask
             is available), AND
          2. no active-threat obstacle's box overlaps that half within the
             forward band.
        With no drivable mask, rule 1 is skipped (assume road exists) and
        only obstacle overlap is checked."""
        x_lo, x_hi = (0, frame_w // 2) if side == "left" else (frame_w // 2, frame_w)
        y_lo, y_hi = int(frame_h * 0.45), frame_h  # forward/near band

        if drivable_mask is not None:
            band = drivable_mask[y_lo:y_hi, x_lo:x_hi]
            if band.size > 0 and not band.any():
                return False  # no road surface on that side at all

        for obs in obstacles:
            if not self._is_threat(obs):
                continue
            ox1, oy1, ox2, oy2 = obs["box"]
            if oy2 < y_lo:
                continue  # threat is far enough away to ignore for corridor check
            if ox2 <= x_lo or ox1 >= x_hi:
                continue  # threat box doesn't overlap this half at all
            return False
        return True

    def _free_side(self, drivable_mask, frame_w: int) -> str:
        """Which half of the drivable area has more pixels overall (used
        only as a tie-breaker when an obstacle is dead-center)."""
        if drivable_mask is None or not drivable_mask.any():
            return "equal"
        left  = int(drivable_mask[:, :frame_w // 2].sum())
        right = int(drivable_mask[:, frame_w // 2:].sum())
        total = max(left + right, 1)
        diff  = (right - left) / total
        if diff > 0.10:
            return "right"
        if diff < -0.10:
            return "left"
        return "equal"

    # ------------------------------------------------------------------
    def predict(self, obstacles, frame_shape, drivable_mask=None) -> str:
        """
        obstacles: list of dicts, each with at least 'box', 'distance',
                   'ttc', 'risk' -- the SAME shape used for vehicles AND
                   animals, so callers just pass vehicles + animals
                   concatenated together.
        Returns 'STRAIGHT', 'LEFT', or 'RIGHT'.
        """
        h, w = frame_shape[:2]
        obstacles = obstacles or []

        left_safe = self._corridor_free("left", drivable_mask, w, h, obstacles)
        right_safe = self._corridor_free("right", drivable_mask, w, h, obstacles)
        self.last_corridor_safety = (left_safe, right_safe)

        active_threats = [o for o in obstacles if self._is_threat(o)]
        if not active_threats:
            self.both_unsafe = False
            return self._vote("STRAIGHT")

        # primary threat = closest active threat
        primary = min(active_threats, key=lambda o: o.get("distance", 999.0))
        lat = self._lateral_zone(primary["box"], w)

        if not left_safe and not right_safe:
            # Both corridors blocked -- never invent a fake-safe direction.
            # Brake logic (elsewhere) handles the actual slowdown.
            self.both_unsafe = True
            return self._vote("STRAIGHT")
        self.both_unsafe = False

        if lat == "center":
            free = self._free_side(drivable_mask, w)
            if right_safe and (free in ("right", "equal") or not left_safe):
                raw = "RIGHT"
            elif left_safe:
                raw = "LEFT"
            else:
                raw = "RIGHT"
        elif lat == "left":
            # threat on the left -> prefer right, but only if actually safe
            raw = "RIGHT" if right_safe else ("LEFT" if left_safe else "STRAIGHT")
        else:
            # threat on the right -> prefer left, but only if actually safe
            raw = "LEFT" if left_safe else ("RIGHT" if right_safe else "STRAIGHT")

        return self._vote(raw)

    def _vote(self, direction: str) -> str:
        """Decay all votes, boost winner, return leading direction."""
        for k in self._votes:
            self._votes[k] *= (1.0 - self._alpha)
        self._votes[direction] += self._alpha
        self.current = max(self._votes, key=self._votes.get)
        return self.current

    # ------------------------------------------------------------------
    def draw_arrow(self, frame, direction: str,
                   drivable_mask=None, risk: str = "SAFE") -> "np.ndarray":
        """
        Draws a circle-arrow indicator inside the drivable area. Kept as a
        compact badge alongside the (now primary) road trajectory drawn by
        trajectory_planner.py. Semi-transparent dark background keeps it
        readable on any scene colour.
        """
        h, w = frame.shape[:2]
        color = _DIR_COLORS[direction]

        if drivable_mask is not None and drivable_mask.any():
            rows = np.where(drivable_mask.any(axis=1))[0]
            bottom_row = int(min(rows.max(), h - 80))
            zone = drivable_mask[max(0, bottom_row - 50):bottom_row + 1, :]
            col_idx = np.where(zone)[1]
            cx = int(np.median(col_idx)) if len(col_idx) > 0 else w // 2
            cy = bottom_row - 40
        else:
            cx, cy = w // 2, h - 100

        cx = int(np.clip(cx, 50, w - 50))
        cy = int(np.clip(cy, 50, h - 50))

        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), 40, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        cv2.circle(frame, (cx, cy), 40, color, 2, cv2.LINE_AA)

        L, TW, TL = 26, 11, 13

        if direction == "STRAIGHT":
            tail, tip = (cx, cy + L), (cx, cy - L)
            cv2.line(frame, tail, tip, color, 4, cv2.LINE_AA)
            pts = np.int32([[cx - TW, cy - L + TL], [cx + TW, cy - L + TL], [cx, cy - L - TL // 2]])
            cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)
        elif direction == "LEFT":
            tail, tip = (cx + L, cy), (cx - L, cy)
            cv2.line(frame, tail, tip, color, 4, cv2.LINE_AA)
            pts = np.int32([[cx - L + TL, cy - TW], [cx - L + TL, cy + TW], [cx - L - TL // 2, cy]])
            cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)
        elif direction == "RIGHT":
            tail, tip = (cx - L, cy), (cx + L, cy)
            cv2.line(frame, tail, tip, color, 4, cv2.LINE_AA)
            pts = np.int32([[cx + L - TL, cy - TW], [cx + L - TL, cy + TW], [cx + L + TL // 2, cy]])
            cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)

        label = _DIR_LABEL[direction] if not self.both_unsafe else "HOLD / BRAKE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        lx, ly = cx - tw // 2, cy + 56
        cv2.rectangle(frame, (lx - 4, ly - th - 2), (lx + tw + 4, ly + 4), (15, 15, 15), -1, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

        return frame
