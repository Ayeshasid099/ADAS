import math
import numpy as np
from collections import defaultdict

# Store history for every tracked object
history = defaultdict(lambda: {
    "speed": [],
    "distance": [],
    "area": [],
    "accel": [],
    "last_time": None,
    "last_center": None
})


def extract_features(track_id, bbox, frame_shape, distance, current_time):
    """
    bbox = (x1,y1,x2,y2)
    frame_shape = frame.shape
    distance = estimated distance (meters)
    """

    x1, y1, x2, y2 = bbox

    frame_h, frame_w = frame_shape[:2]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    area = bbox_w * bbox_h
    area_norm = area / (frame_w * frame_h)

    aspect = bbox_w / max(bbox_h, 1)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    cx_norm = cx / frame_w
    cy_norm = cy / frame_h

    lat_norm = cx_norm - 0.5

    h = history[track_id]

    speed_px = 0
    accel = 0
    jerk = 0
    ttc = 999

    if h["last_center"] is not None:

        prev_cx, prev_cy = h["last_center"]

        dt = current_time - h["last_time"]

        if dt > 0:

            dx = cx - prev_cx
            dy = cy - prev_cy

            speed_px = math.sqrt(dx*dx + dy*dy) / dt

    h["speed"].append(speed_px)

    if len(h["speed"]) > 5:
        h["speed"].pop(0)

    speed_mean5 = np.mean(h["speed"])
    speed_std5 = np.std(h["speed"])

    if len(h["speed"]) >= 2:

        accel = (h["speed"][-1] - h["speed"][-2]) / max(dt, 1e-5)

    h["accel"].append(accel)

    if len(h["accel"]) > 5:
        h["accel"].pop(0)

    if len(h["accel"]) >= 2:

        jerk = (h["accel"][-1] - h["accel"][-2]) / max(dt, 1e-5)

    h["distance"].append(distance)

    if len(h["distance"]) > 5:
        h["distance"].pop(0)

    dist_mean5 = np.mean(h["distance"])

    if len(h["distance"]) >= 2:

        approach_flag = int(h["distance"][-1] < h["distance"][-2])

    else:
        approach_flag = 0

    if speed_mean5 > 0.1:

        ttc = distance / speed_mean5

    h["area"].append(area)

    if len(h["area"]) > 5:
        h["area"].pop(0)

    if len(h["area"]) >= 2:

        expand_ratio = h["area"][-1] / max(h["area"][-2], 1)

        size_change = (
            h["area"][-1] - h["area"][-2]
        ) / max(h["area"][-2], 1)

    else:

        expand_ratio = 1.0
        size_change = 0.0

    h["last_center"] = (cx, cy)
    h["last_time"] = current_time

    features = np.array([[
        bbox_h,
        bbox_w,
        area_norm,
        aspect,
        cx_norm,
        cy_norm,
        speed_px,
        speed_mean5,
        speed_std5,
        accel,
        jerk,
        distance,
        dist_mean5,
        ttc,
        expand_ratio,
        size_change,
        lat_norm,
        approach_flag
    ]], dtype=np.float32)

    return features