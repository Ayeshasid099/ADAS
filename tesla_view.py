"""
tesla_view.py
------------------
"Debug visualization" render mode: the frame goes black & white, then
everything NOT actually detected gets smoothed over with classical
inpainting (cv2.inpaint) instead of staying sharp raw footage. Whatever
ADAS actually detected (vehicles, potholes, humps, lane, drivable area)
then gets drawn crisp and in color ON TOP of that painted background --
similar in spirit to Tesla FSD's visualizer, which also drops raw camera
detail and only renders what the model is confident about.

IMPORTANT — what "AI" means here: cv2.inpaint (Telea algorithm) is a
classical computer-vision interpolation method, not a trained neural
network. A real generative/diffusion inpainting model cannot run
per-frame at video framerate on consumer hardware — a single frame alone
takes several seconds on an RTX 3050. This is the practical real-time
substitute, as discussed.

Usage (inside the main frame loop, after all 5 models have run):

    from tesla_view import build_known_mask, render_painted_background

    known = build_known_mask(
        frame.shape,
        vehicle_boxes=[v["box"] for v in vehicles],
        pothole_boxes=potholes,
        hump_boxes=humps,
        lane_mask=lane_mask,
        drivable_mask=drivable_mask,
    )
    frame = render_painted_background(frame, known, inpaint_radius=8, downscale=0.5)
    # ... then draw all the normal colored overlays/boxes on `frame` as before
"""

import cv2
import numpy as np


def build_known_mask(frame_shape, vehicle_boxes=None, pothole_boxes=None,
                      hump_boxes=None, lane_mask=None, drivable_mask=None,
                      box_pad=6):
    """Boolean (H, W) mask: True wherever something was actually detected
    this frame (vehicle/person box, pothole, hump, lane pixel, drivable-
    area pixel). Everything False is 'unknown' background that gets
    painted over instead of shown as sharp raw footage."""
    h, w = frame_shape[:2]
    box_mask = np.zeros((h, w), dtype=np.uint8)

    for boxes in (vehicle_boxes, pothole_boxes, hump_boxes):
        if not boxes:
            continue
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, int(x1) - box_pad)
            y1 = max(0, int(y1) - box_pad)
            x2 = min(w, int(x2) + box_pad)
            y2 = min(h, int(y2) + box_pad)
            box_mask[y1:y2, x1:x2] = 255

    known = box_mask.astype(bool)
    if lane_mask is not None:
        known |= lane_mask
    if drivable_mask is not None:
        known |= drivable_mask
    return known


def render_painted_background(frame, known_mask, inpaint_radius=8, downscale=0.5):
    """Grayscale the frame, then classically inpaint every pixel NOT
    covered by `known_mask` so the background reads as a soft painted
    backdrop instead of sharp raw footage.

    Runs downscaled then resizes back up: cv2.inpaint's cost scales with
    the masked area, and most of a frame is usually 'unknown', so
    full-resolution inpainting is too slow for a live 30fps loop.
    `downscale=0.5` keeps it comfortably real-time; drop to 0.35 if your
    machine still lags."""
    h, w = frame.shape[:2]
    small_w, small_h = max(1, int(w * downscale)), max(1, int(h * downscale))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    small_frame = cv2.resize(gray_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_known = cv2.resize(known_mask.astype(np.uint8) * 255, (small_w, small_h),
                              interpolation=cv2.INTER_NEAREST)

    fill_mask = 255 - small_known  # everything NOT known gets painted over
    painted_small = cv2.inpaint(small_frame, fill_mask, inpaint_radius, cv2.INPAINT_TELEA)

    painted = cv2.resize(painted_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return painted
