"""
projection_3d.py
------------------
STAGE 2 of the 3D-engine roadmap: turn a 2D pixel + a distance into a real
3D world coordinate, and calibrate MiDaS's relative depth map against your
EXISTING metric distance estimates so background/undetected pixels can be
placed in 3D too.

Two separate things happen here:

  1. PINHOLE PROJECTION (project_point / project_box_ground_point)
     For anything you already have a trustworthy metric distance for --
     i.e. every vehicle/person your main model detects, via the existing
     `real_width * FOCAL_LENGTH / pixel_width` math in
     adas_full_pipeline.py -- this does NOT need MiDaS at all. Standard
     pinhole camera geometry recovers lateral offset (X) and height (Z)
     directly from the pixel position + known forward distance (Y).

  2. DEPTH CALIBRATION (DepthCalibrator)
     MiDaS only gives RELATIVE depth (near/far ordering), not meters. To
     place the ROAD SURFACE, CURBS, or anything undetected in 3D, we fit a
     simple linear relationship between MiDaS's raw values and the metric
     distances you already trust (from #1, at the same pixel locations)
     every frame you have vehicles in view, then use that fit to convert
     any other MiDaS pixel to an approximate metric distance.

World coordinate convention used everywhere below (right-handed, matches
how you'd want a bird's-eye/chase-cam render to look):
    X : lateral offset, meters, POSITIVE = right of camera
    Y : forward distance, meters, POSITIVE = further from camera
    Z : height, meters, POSITIVE = above the camera's own height

Usage:

    from projection_3d import DepthCalibrator, project_box_ground_point

    calibrator = DepthCalibrator()

    # every frame, for each vehicle you already trust the distance of:
    calibrator.add_anchor(midas_region_depth, vehicle_metric_distance)

    # every ~30 frames, refit once you've collected a few distinct anchors:
    calibrator.fit()

    # place a KNOWN-distance detection in 3D (no MiDaS needed):
    X, Y, Z = project_box_ground_point(box, metric_distance, frame.shape, FOCAL_LENGTH)

    # place an UNKNOWN pixel (e.g. road surface) in 3D using calibrated MiDaS:
    metric_dist = calibrator.to_metric(midas_value_at_that_pixel)
    if metric_dist is not None:
        X, Y, Z = project_point(u, v, metric_dist, frame.shape, FOCAL_LENGTH)
"""

from collections import deque

import numpy as np


def project_point(u, v, depth_m, frame_shape, focal_length):
    """Pinhole projection: pixel (u, v) + forward distance depth_m (meters)
    -> 3D world point (X, Y, Z) in meters. Assumes square pixels (fx=fy=
    focal_length, same constant you already calibrated in
    adas_full_pipeline.py) and principal point at the frame center."""
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    X = (u - cx) * depth_m / focal_length
    Y = depth_m
    Z = -(v - cy) * depth_m / focal_length  # image v grows downward -> flip sign for "up"

    return X, Y, Z


def unproject_point(X, Y, Z, frame_shape, focal_length):
    """Inverse of project_point: 3D world point -> pixel (u, v). Used only
    to unit-test project_point's correctness below, and later in Stage 3
    if you need to check whether a 3D point is on-screen."""
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    u = cx + (X * focal_length) / Y
    v = cy - (Z * focal_length) / Y
    return u, v


def project_box_ground_point(box, depth_m, frame_shape, focal_length):
    """Convenience wrapper for a detection box (x1, y1, x2, y2): projects
    the BOTTOM-CENTER pixel (the point most likely touching the road) at
    the given metric depth_m. This is what you want for placing a 3D car
    icon on the ground plane in Stage 3, not the box center."""
    x1, y1, x2, y2 = box
    u = (x1 + x2) / 2.0
    v = y2
    return project_point(u, v, depth_m, frame_shape, focal_length)


class DepthCalibrator:
    """Fits MiDaS's relative depth values to your trusted metric distances
    (from REAL_WIDTHS-based vehicle distances), so pixels with NO known
    class/width -- road surface, curbs, unclassified obstacles -- can still
    get a metric depth via the fitted relationship.

    MiDaS output is roughly proportional to INVERSE depth (disparity), so
    the fit is: midas_value ~= a * (1 / metric_distance) + b
    which inverts to: metric_distance = a / (midas_value - b)
    """

    def __init__(self, max_anchors=300, min_anchors_to_fit=6):
        self.midas_vals = deque(maxlen=max_anchors)
        self.metric_dists = deque(maxlen=max_anchors)
        self.min_anchors_to_fit = min_anchors_to_fit
        self.a = None
        self.b = None

    def add_anchor(self, midas_value, metric_distance):
        """Call once per trusted detection per frame (e.g. every vehicle
        box, using its MiDaS region_depth() value and its REAL_WIDTHS-based
        metric distance). Skips degenerate points that would break the fit."""
        if metric_distance is None or metric_distance <= 0.1:
            return
        if midas_value is None or not np.isfinite(midas_value):
            return
        self.midas_vals.append(float(midas_value))
        self.metric_dists.append(float(metric_distance))

    def fit(self):
        """Refit the linear relationship from all accumulated anchors.
        Returns True if the fit succeeded (enough distinct-enough points),
        False if there's not enough data yet -- keep using the previous
        fit (or none) until then."""
        n = len(self.metric_dists)
        if n < self.min_anchors_to_fit:
            return False

        inv_dist = 1.0 / np.array(self.metric_dists, dtype=np.float64)
        midas = np.array(self.midas_vals, dtype=np.float64)

        # need actual spread in distances, or the fit is meaningless
        if np.std(inv_dist) < 1e-6:
            return False

        a, b = np.polyfit(inv_dist, midas, 1)
        if abs(a) < 1e-9:
            return False  # degenerate slope, would blow up on inversion

        self.a, self.b = float(a), float(b)
        return True

    def to_metric(self, midas_value):
        """Convert a raw MiDaS value to an approximate metric distance
        (meters) using the current fit. Returns None if not calibrated yet
        or the result would be nonsensical (non-positive distance)."""
        if self.a is None or midas_value is None or not np.isfinite(midas_value):
            return None
        denom = midas_value - self.b
        if abs(denom) < 1e-9:
            return None
        dist = self.a / denom
        if dist <= 0 or dist > 500:  # sanity bound, tune for your scenes
            return None
        return dist

    @property
    def is_calibrated(self):
        return self.a is not None
