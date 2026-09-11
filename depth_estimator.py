"""
depth_estimator.py
------------------
STAGE 1 of the 3D-engine roadmap: monocular depth estimation.

Your existing distance math (`real_width * FOCAL_LENGTH / pixel_width` in
adas_full_pipeline.py) only gives a distance for boxes YOLO detected, and
only for classes with a known real-world width. It says nothing about the
road, curbs, or anything undetected.

MiDaS instead predicts a depth value for EVERY pixel in the frame from a
single camera image. That's what Stage 2 (2D -> 3D projection) needs to
place both detected objects AND the general road surface in 3D space.

NOTE: MiDaS output is RELATIVE depth (near/far ordering + rough shape),
not metric meters, out of the box. Stage 2 will calibrate it against your
known-class distances (person/car/etc via REAL_WIDTHS) so the two systems
agree on a common scale.

Model choice: MiDaS_small — smallest/fastest variant. Still not real-time
on CPU (few fps at best), which is why this whole pipeline is an OFFLINE
batch process (Stage 4), not something bolted into the live cv2 loop.

Usage:
    from depth_estimator import DepthEstimator

    de = DepthEstimator()          # loads model once, ~80MB download first run
    depth_map = de.predict(frame)  # frame: BGR np.uint8 (H,W,3) -> depth_map: float32 (H,W)

    # depth_map values: HIGHER = CLOSER (MiDaS convention: inverse depth).
    # Use de.depth_at(depth_map, x, y) to sample it at a pixel.
"""

import cv2
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[depth_estimator] loading {model_type} on {self.device} "
              f"(first run downloads the weights, ~80MB)...")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # small_transform matches MiDaS_small / DPT_Hybrid; use default_transform
        # instead if you switch to "DPT_Large" for higher quality later.
        self.transform = midas_transforms.small_transform

        print("[depth_estimator] ready.")

    def predict(self, frame_bgr):
        """frame_bgr: np.uint8 (H, W, 3) in BGR (straight from cv2.VideoCapture).
        Returns: np.float32 (H, W) relative depth map, same resolution as
        the input frame. HIGHER value = CLOSER to the camera."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy().astype(np.float32)

    @staticmethod
    def depth_at(depth_map, x, y):
        """Sample the depth map at a pixel, clamped to bounds."""
        h, w = depth_map.shape[:2]
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        return float(depth_map[yi, xi])

    @staticmethod
    def region_depth(depth_map, box, agg="median"):
        """Depth over a box region (x1,y1,x2,y2) instead of one pixel --
        more stable for a whole vehicle than a single point sample.
        agg: 'median' (robust to edge/background pixels in the box) or 'mean'."""
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = depth_map.shape[:2]
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = depth_map[y1:y2, x1:x2]
        return float(np.median(region) if agg == "median" else np.mean(region))

    @staticmethod
    def colorize(depth_map):
        """Turn a raw depth map into a viewable BGR heatmap (for debugging
        / sanity-checking what the model actually sees)."""
        d = depth_map.copy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        d = (d * 255).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)
