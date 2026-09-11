"""
seg_utils.py
------------------
Shared helpers for the two YOLOv8-SEGMENTATION models in the pipeline:

    - drivable-area model  (bdd100k_40k_yolov8n_seg_results/weights/best.pt)
    - lane / road-line model (linerodebest.pt)

Both output r.masks (per-instance masks at model input resolution), not
boxes-only, so they need different drawing logic than the detection
models (pothole/hump/vehicles).

Usage:
    from seg_utils import build_mask, blend_mask

    drivable_mask = build_mask(drivable_results, frame.shape)
    frame = blend_mask(frame, drivable_mask, color=(0, 200, 0), alpha=0.30)

    lane_mask = build_mask(lane_results, frame.shape, class_filter={0, 1})
    frame = blend_mask(frame, lane_mask, color=(0, 255, 255), alpha=0.55)
"""

import cv2
import numpy as np


def build_mask(results, frame_shape, class_filter=None):
    """Collapse every instance mask in a YOLOv8-seg `results` list into a
    single boolean (H, W) mask, resized to the ORIGINAL frame size.

    class_filter : optional set/list of class ids to keep (e.g. only the
                    "drivable area" class if the seg model has more than
                    one class). None = keep everything the model found.
    """
    h, w = frame_shape[:2]
    combined = np.zeros((h, w), dtype=bool)

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()  # (N, mh, mw), values 0..1
        cls_ids = (
            r.boxes.cls.cpu().numpy().astype(int)
            if r.boxes is not None and r.boxes.cls is not None
            else [None] * len(masks)
        )

        for m, cid in zip(masks, cls_ids):
            if class_filter is not None and cid not in class_filter:
                continue
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            combined |= (m_resized > 0.5)

    return combined


def blend_mask(frame, mask, color=(0, 255, 0), alpha=0.35):
    """Alpha-blend a solid `color` onto `frame` wherever `mask` is True.
    Cheap: builds the colored layer once and blends only the masked
    pixels, so stacking drivable-area + lane overlays doesn't double-darken
    areas that only one of them covers."""
    if not mask.any():
        return frame

    colored = np.empty_like(frame)
    colored[:] = color
    blended = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
    frame[mask] = blended[mask]
    return frame


def mask_outline(mask, color=(0, 255, 255), thickness=2):
    """Return contour points for drawing a crisp border around a mask
    (useful for the drivable-area boundary so it doesn't look like a flat
    green blob with no edge definition)."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
