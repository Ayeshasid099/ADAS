"""
adas_core.py
------------------
Shared ADAS engine, extracted from adas_full_pipeline.py so the SAME
processing code (models, distance/speed/TTC math, brake, direction,
trajectory, animal detection) can be reused by:

  - adas_full_pipeline.py  (desktop cv2.imshow app)
  - adas_engine.py         (Flask web app: video upload + live camera)

Nothing in here opens a cv2 window or calls cv2.imshow -- that stays in
the two callers above. This module only: loads models, processes one
frame at a time, and draws the annotated overlay onto that frame.

v1 additions vs the old inline pipeline script:
  - Animal detection (animal_best.pt), fully wired into the SAME
    obstacle/risk/brake/direction/trajectory/logging/3D pipeline that
    vehicles already use -- no parallel algorithm.
  - Real corridor-safety direction guidance (direction_predictor v3).
  - trajectory_planner.py road-path overlay.
  - Reusable AdasModels / AdasSession so a Flask request can spin up an
    isolated per-session state without stepping on module-level globals
    (the old script used bare module-level variables, which only works
    for a single always-running desktop process).
"""

from __future__ import annotations

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

from feature_extractor import extract_features
from collision_model import CollisionModel
from brake_predictor import BrakePredictor
from direction_predictor import DirectionPredictor
from trajectory_planner import TrajectoryPlanner
from seg_utils import build_mask, blend_mask
from tesla_view import build_known_mask, render_painted_background

# ============================================================
# 1. MODEL PATHS  (>>> fix these for your setup <<<)
# ============================================================
MAIN_MODEL_PATH     = "modals/car_best.pt"
POTHOLE_MODEL_PATH  = "modals/bestphotole.pt"
HUMP_MODEL_PATH      = "modals/besthumb.pt"
LANE_MODEL_PATH      = "modals/linerodebest.pt"
DRIVABLE_MODEL_PATH  = "bdd100k_40k_yolov8n_seg_results/weights/best.pt"
COLLISION_MODEL_PATH = "modals/best_model.pkl"
ANIMAL_MODEL_PATH    = "modals/animal_best.pt"

# ============================================================
# 2. THRESHOLDS / CALIBRATION
# ============================================================
POTHOLE_CONF, POTHOLE_IOU = 0.40, 0.50
HUMP_CONF,    HUMP_IOU    = 0.40, 0.50
LANE_CONF                 = 0.35
DRIVABLE_CONF             = 0.35
ANIMAL_CONF, ANIMAL_IOU   = 0.35, 0.45
DETECT_EVERY_N = 2
SEG_EVERY_N    = 3

FOCAL_LENGTH    = 800
CAMERA_HEIGHT_M = 1.30

REAL_WIDTHS = {
    "person": 0.5, "rider": 0.6, "car": 1.8,
    "bus": 2.5, "truck": 2.5, "bike": 0.7, "motor": 0.7,
}

# Configurable animal real-world width mapping (meters), with a safe
# fallback for any class name the animal model outputs that isn't
# listed here. model.names is read DYNAMICALLY at load time -- we never
# hard-code which animal classes exist.
DEFAULT_ANIMAL_WIDTH = 0.5
ANIMAL_WIDTHS = {
    "dog": 0.35, "cat": 0.20, "cow": 1.3, "goat": 0.4, "sheep": 0.5,
    "horse": 1.1, "deer": 0.6, "buffalo": 1.5, "camel": 1.2, "pig": 0.5,
    "elephant": 2.5, "monkey": 0.3,
}

CLASS_COLORS = {
    "car":           (255, 204, 102),
    "truck":         (102, 178, 255),
    "bus":           (204, 153, 255),
    "person":        (153, 255, 153),
    "rider":         (255, 153, 204),
    "bike":          (153, 255, 255),
    "motor":         (153, 204, 255),
    "pothole":       (255, 102, 255),
    "hump":          (0, 255, 255),
    "traffic sign":  (102, 255, 255),
    "traffic light": (102, 255, 255),
    "animal":        (0, 220, 220),
}
DEFAULT_ANIMAL_COLOR = (60, 220, 255)

DRIVABLE_COLOR = (0, 200, 0)
LANE_COLOR     = (0, 0, 255)

RISK_COLOR = {"DANGER": (0, 0, 255), "WARNING": (0, 165, 255), "SAFE": (0, 255, 0)}


# ============================================================
# 3. POTHOLE / HUMP ground-distance helpers
# ============================================================
def ground_distance(y_bottom: float, frame_h: int, box_height_px: float | None = None):
    horizon_y = frame_h / 2.0
    dy = y_bottom - horizon_y
    if dy >= 5:
        dist = (CAMERA_HEIGHT_M * FOCAL_LENGTH) / dy
        if 0.5 <= dist <= 80:
            return dist, False
    if box_height_px and box_height_px > 1:
        ASSUMED_FEATURE_HEIGHT_M = 0.12
        dist = (ASSUMED_FEATURE_HEIGHT_M * FOCAL_LENGTH) / box_height_px
        dist = max(1.0, min(dist, 60.0))
        return dist, True
    return None, False


def ground_size_cm(x1, y1, x2, y2, dist_m: float) -> tuple[int, int]:
    px_w, px_h = x2 - x1, y2 - y1
    w_m = (px_w * dist_m) / FOCAL_LENGTH
    h_m = (px_h * dist_m) / FOCAL_LENGTH
    return max(1, round(w_m * 100)), max(1, round(h_m * 100))


# ============================================================
# 4. Generic IoU-based smooth detector (potholes / humps)
# ============================================================
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


class SmoothDetector:
    def __init__(self, max_age=6, iou_thresh=0.35):
        self.tracks = []
        self.max_age = max_age
        self.iou_thresh = iou_thresh

    def update(self, new_boxes):
        matched = set()
        for nb in new_boxes:
            best_iou, best_idx = 0.0, -1
            for i, t in enumerate(self.tracks):
                if i in matched:
                    continue
                iou = _iou(nb, t['box'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx != -1 and best_iou > self.iou_thresh:
                self.tracks[best_idx]['box'] = nb
                self.tracks[best_idx]['age'] = 0
                matched.add(best_idx)
            else:
                self.tracks.append({'box': nb, 'age': 0})
        survivors = []
        for i, t in enumerate(self.tracks):
            if i not in matched:
                t['age'] += 1
            if t['age'] <= self.max_age:
                survivors.append(t)
        self.tracks = survivors
        return [t['box'] for t in self.tracks]


def label_from_score(score: float) -> str:
    if score >= 0.60:
        return "DANGER"
    elif score >= 0.25:
        return "WARNING"
    return "SAFE"


def kinematics_from_track(track_id, raw_distance, current_time, history: dict):
    """SAME smoothing formula the original vehicle pipeline used (EMA
    distance + EMA speed -> TTC), factored out so animals reuse it
    identically instead of a second algorithm.
    Returns (final_distance, speed_kmh, ttc)."""
    speed_kmh, ttc = 0.0, float("inf")
    if track_id in history:
        prev = history[track_id]
        sm_d = prev["dist"] * 0.8 + raw_distance * 0.2
        dt = current_time - prev["time"]
        sm_spd = 0.0
        if dt > 0:
            raw_spd_ms = (prev["dist"] - sm_d) / dt
            sm_spd = prev.get("speed", 0) * 0.8 + raw_spd_ms * 0.2
            if sm_spd > 0.3:
                speed_kmh = sm_spd * 3.6
                ttc = sm_d / sm_spd
        final_distance = sm_d
        history[track_id] = {"dist": sm_d, "time": current_time, "speed": sm_spd}
    else:
        final_distance = raw_distance
        history[track_id] = {"dist": raw_distance, "time": current_time, "speed": 0}
    return final_distance, speed_kmh, ttc


# ============================================================
# 5. Model bundle (load once, shared across sessions)
# ============================================================
class AdasModels:
    def __init__(self, base_dir: str = "."):
        p = lambda rel: os.path.join(base_dir, rel)

        self.model          = YOLO(p(MAIN_MODEL_PATH))
        self.pothole_model  = YOLO(p(POTHOLE_MODEL_PATH))
        self.hump_model     = YOLO(p(HUMP_MODEL_PATH))
        self.lane_model     = YOLO(p(LANE_MODEL_PATH))
        self.drivable_model = YOLO(p(DRIVABLE_MODEL_PATH))
        self.collision_model = CollisionModel(p(COLLISION_MODEL_PATH))

        # Animal model is OPTIONAL -- missing/broken weights must never
        # take down the rest of ADAS.
        self.animal_model = None
        try:
            self.animal_model = YOLO(p(ANIMAL_MODEL_PATH))
            print(f"[adas_core] animal model loaded: {self.animal_model.names}")
        except Exception as e:
            print(f"[adas_core] WARNING: could not load animal model "
                  f"'{ANIMAL_MODEL_PATH}' ({e}). Continuing WITHOUT animal "
                  f"detection -- vehicle ADAS is unaffected.")

        print("[main model]     ", self.model.names)
        print("[lane model]     ", self.lane_model.names)
        print("[drivable model] ", self.drivable_model.names)


# ============================================================
# 6. Per-session mutable state
# ============================================================
class AdasSession:
    """Everything that needs to persist frame-to-frame for ONE video/
    camera run. Isolated per session so Flask can run multiple
    video/camera sessions without cross-talk (the old script used bare
    module-level globals, fine for one desktop process, not for a
    multi-request web server)."""

    def __init__(self, tesla_mode: bool = False):
        self.pothole_tracker = SmoothDetector(max_age=6, iou_thresh=0.35)
        self.hump_tracker    = SmoothDetector(max_age=6, iou_thresh=0.35)

        self.last_drivable_mask = None
        self.last_lane_mask     = None

        self.track_history        = {}   # vehicles
        self.risk_score_history   = {}   # vehicles (ML score EMA)
        self.animal_track_history = {}   # animals
        self.animal_score_history = {}   # animals (heuristic score EMA)

        self.brake_pred = BrakePredictor(alpha_rise=0.18, alpha_fall=0.25)
        self.dir_pred   = DirectionPredictor(alpha=0.22)
        self.trajectory = TrajectoryPlanner(alpha=0.20)

        self.tesla_mode = tesla_mode
        self.frame_count = 0
        self.session_start = time.time()

        # optional, wired up by the caller (desktop script / Flask engine)
        self.data_logger = None
        self.web_reporter = None


# ============================================================
# 7. Per-frame processing (no drawing)
# ============================================================
def process_frame(models: AdasModels, session: AdasSession, frame,
                   current_time: float | None = None) -> dict:
    """Runs every model + all the risk/brake/direction/trajectory math
    for one frame. Returns a plain dict -- no cv2 drawing here, so this
    same function works for both the desktop loop and Flask (offline
    video processing AND live MJPEG streaming)."""
    current_time = current_time if current_time is not None else time.time()
    session.frame_count += 1
    frame_count = session.frame_count
    h, w = frame.shape[:2]
    session_t = current_time - session.session_start

    results = models.model.track(frame, persist=True, conf=0.35, iou=0.45,
                                  tracker="bytetrack.yaml", verbose=False)

    animal_results = []
    if models.animal_model is not None:
        try:
            animal_results = models.animal_model.track(
                frame, persist=True, conf=ANIMAL_CONF, iou=ANIMAL_IOU,
                tracker="bytetrack.yaml", verbose=False)
        except Exception as e:
            print(f"[adas_core] animal model inference failed this frame: {e}")
            animal_results = []

    run_det = (frame_count % DETECT_EVERY_N == 0)
    run_seg = (frame_count % SEG_EVERY_N == 0)

    raw_potholes, raw_humps = [], []
    if run_det:
        for r in models.pothole_model(frame, conf=POTHOLE_CONF, iou=POTHOLE_IOU, verbose=False):
            for box in r.boxes:
                raw_potholes.append(tuple(map(float, box.xyxy[0])))
        for r in models.hump_model(frame, conf=HUMP_CONF, iou=HUMP_IOU, verbose=False):
            for box in r.boxes:
                raw_humps.append(tuple(map(float, box.xyxy[0])))

    potholes = session.pothole_tracker.update(raw_potholes)
    humps    = session.hump_tracker.update(raw_humps)

    if run_seg:
        drivable_results = models.drivable_model(frame, conf=DRIVABLE_CONF, verbose=False)
        session.last_drivable_mask = build_mask(drivable_results, frame.shape)
        lane_results = models.lane_model(frame, conf=LANE_CONF, verbose=False)
        session.last_lane_mask = build_mask(lane_results, frame.shape)

    drivable_mask = session.last_drivable_mask
    lane_mask     = session.last_lane_mask

    vehicles, signs = [], []

    # ---- vehicles / persons / signs ----
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = models.model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name in ("pothole", "hump", "lane", "drivable area"):
                continue
            if name in ("traffic sign", "traffic light"):
                signs.append((name, x1, y1, x2, y2))
                continue
            if name not in REAL_WIDTHS:
                continue

            pixel_width = max(x2 - x1, 1)
            raw_distance = (REAL_WIDTHS[name] * FOCAL_LENGTH) / pixel_width

            speed_kmh, ttc, risk, risk_score = 0.0, float("inf"), "SAFE", 0.0
            final_distance = raw_distance

            if box.id is not None:
                track_id = int(box.id[0])
                final_distance, speed_kmh, ttc = kinematics_from_track(
                    track_id, raw_distance, current_time, session.track_history)

                features = extract_features(
                    track_id, (x1, y1, x2, y2), frame.shape, final_distance, current_time)

                raw_score = models.collision_model.predict_proba(features[0])
                prev_score = session.risk_score_history.get(track_id, raw_score)
                risk_score = 0.35 * raw_score + 0.65 * prev_score
                session.risk_score_history[track_id] = risk_score
                risk = label_from_score(risk_score)

            if final_distance > 12:
                risk, risk_score = "SAFE", min(risk_score, 0.05)
            elif final_distance > 5 and risk == "DANGER":
                risk, risk_score = "WARNING", min(risk_score, 0.55)

            vehicles.append({
                "kind": "vehicle", "mesh_kind": name,
                "name": name, "box": (x1, y1, x2, y2),
                "distance": final_distance, "speed": speed_kmh, "ttc": ttc,
                "class_color": CLASS_COLORS.get(name, (255, 255, 255)),
                "risk_color": RISK_COLOR[risk], "risk": risk, "risk_score": risk_score,
            })

    # ---- animals (reuses the exact same distance/speed/TTC/risk shape) ----
    animals = []
    for r in animal_results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        names_map = models.animal_model.names  # read dynamically, never hard-coded
        for box in boxes:
            cls = int(box.cls[0])
            name = names_map.get(cls, f"animal_{cls}") if isinstance(names_map, dict) else names_map[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            real_w = ANIMAL_WIDTHS.get(name, DEFAULT_ANIMAL_WIDTH)
            pixel_width = max(x2 - x1, 1)
            raw_distance = (real_w * FOCAL_LENGTH) / pixel_width

            speed_kmh, ttc = 0.0, float("inf")
            final_distance = raw_distance

            if box.id is not None:
                track_id = int(box.id[0])
                final_distance, speed_kmh, ttc = kinematics_from_track(
                    track_id, raw_distance, current_time, session.animal_track_history)
            else:
                track_id = None

            # No dedicated ML model for animals -- a transparent distance/
            # TTC heuristic score, same 0-1 scale + same EMA smoothing
            # approach as the vehicle ML score, so it blends into brake/
            # direction identically.
            dist_component = max(0.0, min(1.0, (12.0 - final_distance) / 12.0))
            ttc_component = 0.0 if ttc == float("inf") else max(0.0, min(1.0, (4.0 - ttc) / 4.0))
            raw_score = max(dist_component, ttc_component)
            hist_key = track_id if track_id is not None else id(box)
            prev_score = session.animal_score_history.get(hist_key, raw_score)
            risk_score = 0.35 * raw_score + 0.65 * prev_score
            session.animal_score_history[hist_key] = risk_score
            risk = label_from_score(risk_score)

            if final_distance > 12:
                risk, risk_score = "SAFE", min(risk_score, 0.05)
            elif final_distance > 5 and risk == "DANGER":
                risk, risk_score = "WARNING", min(risk_score, 0.55)

            animals.append({
                "kind": "animal", "mesh_kind": "animal",
                "name": name, "box": (x1, y1, x2, y2),
                "distance": final_distance, "speed": speed_kmh, "ttc": ttc,
                "class_color": CLASS_COLORS.get("animal", DEFAULT_ANIMAL_COLOR),
                "risk_color": RISK_COLOR[risk], "risk": risk, "risk_score": risk_score,
            })

    obstacles = vehicles + animals

    if session.data_logger is not None:
        session.data_logger.log(frame_count, current_time, obstacles)

    closest = min(obstacles, key=lambda o: o["distance"], default=None)
    risk_now = closest["risk"] if closest else "SAFE"

    # ---- brake ----
    if closest:
        brake_pct, brake_col = session.brake_pred.update(
            closest["distance"], closest["ttc"], closest["risk"],
            speed_kmh=closest["speed"], risk_score=closest.get("risk_score"))
    else:
        brake_pct, brake_col = session.brake_pred.update(
            999.0, float("inf"), "SAFE", speed_kmh=0.0, risk_score=0.0)

    if session.web_reporter is not None:
        session.web_reporter.log_brake(session_t, brake_pct)

    # ---- direction (obstacle + corridor aware) ----
    direction = session.dir_pred.predict(obstacles, frame.shape, drivable_mask)

    # ---- trajectory ----
    trajectory_pts = session.trajectory.compute_safe_trajectory(
        direction, drivable_mask, frame.shape,
        obstacle_boxes=[o["box"] for o in obstacles])

    return {
        "frame_count": frame_count, "session_t": session_t,
        "vehicles": vehicles, "animals": animals, "obstacles": obstacles,
        "potholes": potholes, "humps": humps, "signs": signs,
        "drivable_mask": drivable_mask, "lane_mask": lane_mask,
        "closest": closest, "risk_now": risk_now,
        "brake_pct": brake_pct, "brake_col": brake_col,
        "direction": direction, "trajectory_pts": trajectory_pts,
        "corridor_safety": session.dir_pred.last_corridor_safety,
    }


# ============================================================
# 8. Drawing (annotates a frame in place, returns it)
# ============================================================
def draw_overlay(frame, result: dict, session: AdasSession):
    h, w = frame.shape[:2]

    if session.tesla_mode:
        known_mask = build_known_mask(
            frame.shape,
            vehicle_boxes=[o["box"] for o in result["obstacles"]],
            pothole_boxes=result["potholes"], hump_boxes=result["humps"],
            lane_mask=result["lane_mask"], drivable_mask=result["drivable_mask"])
        frame = render_painted_background(frame, known_mask, inpaint_radius=8, downscale=0.5)

    if result["drivable_mask"] is not None:
        frame = blend_mask(frame, result["drivable_mask"], color=DRIVABLE_COLOR, alpha=0.28)
    if result["lane_mask"] is not None:
        frame = blend_mask(frame, result["lane_mask"], color=LANE_COLOR, alpha=0.55)

    # Road-path trajectory overlay only. The circle+arrow "steer left/right"
    # badge (session.dir_pred.draw_arrow) has been removed from the HUD --
    # the underlying corridor-safety logic in direction_predictor.py (and
    # its use in animal/vehicle collision detection + braking) is untouched,
    # only the on-screen arrow indicator is gone.
    frame = session.trajectory.draw_safe_trajectory(
        frame, result["trajectory_pts"], risk=result["risk_now"])

    overlay = frame.copy()
    card_h = 85
    for o in result["obstacles"]:
        x1, y1, x2, y2 = o["box"]
        card_y1 = max(0, y1 - card_h)
        cv2.rectangle(overlay, (x1, card_y1), (x1 + 170, card_y1 + card_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (10, 10), (245, 165), (15, 15, 15), -1)

    bar_w, bar_h = 420, 22
    bar_x, bar_y = w // 2 - bar_w // 2, h - 55
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
    fill_w = int((result["brake_pct"] / 100.0) * bar_w)
    if fill_w > 0:
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), result["brake_col"], -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # ---- potholes / humps ----
    for label, boxes, color_key in (("POTHOLE", result["potholes"], "pothole"),
                                     ("HUMP", result["humps"], "hump")):
        color = CLASS_COLORS[color_key]
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            dist_m, is_est = ground_distance(y2, h, box_height_px=(y2 - y1))
            if dist_m is not None:
                w_cm, h_cm = ground_size_cm(x1, y1, x2, y2, dist_m)
                tag = "~" if is_est else ""
                for line, yoff in ((f"{label}  {tag}{dist_m:.1f} m", -24),
                                    (f"Size: {tag}{w_cm}x{h_cm} cm", -8)):
                    (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1 - 2, y1 + yoff - th - 2), (x1 + tw + 4, y1 + yoff + 4),
                                  (15, 15, 15), -1)
                    cv2.putText(frame, line, (x1, y1 + yoff), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    for (name, x1, y1, x2, y2) in result["signs"]:
        s_color = CLASS_COLORS.get(name, (102, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), s_color, 2, cv2.LINE_AA)
        cv2.putText(frame, name.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 2, cv2.LINE_AA)

    # ---- vehicle + animal cards ----
    for o in result["obstacles"]:
        x1, y1, x2, y2 = o["box"]
        card_y1 = max(0, y1 - card_h)
        speed_str = f"{o['speed']:.1f} km/h" if o["speed"] > 0 else "-- km/h"
        ttc_str = f"{o['ttc']:.1f} s" if o["ttc"] != float("inf") else "-- s"

        cv2.rectangle(frame, (x1, y1), (x2, y2), o["class_color"], 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, card_y1), (x1 + 170, card_y1 + card_h), o["class_color"], 1, cv2.LINE_AA)
        tag = " [ANIMAL]" if o["kind"] == "animal" else ""
        cv2.putText(frame, (o["name"].upper() + tag), (x1 + 8, card_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, o["class_color"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"Risk : {o['risk']}", (x1 + 8, card_y1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, o["risk_color"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Dist : {o['distance']:.1f} m", (x1 + 8, card_y1 + 49),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Speed: {speed_str}", (x1 + 8, card_y1 + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"TTC  : {ttc_str}", (x1 + 8, card_y1 + 79),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- HUD status panel ----
    cv2.rectangle(frame, (10, 10), (245, 165), (255, 255, 255), 1, cv2.LINE_AA)
    total_objects = len(result["obstacles"]) + len(result["potholes"]) + len(result["humps"]) + len(result["signs"])
    cv2.putText(frame, "ADAS STATUS", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objects : {total_objects}", (25, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    lane_status = "OK" if (result["lane_mask"] is not None and result["lane_mask"].any()) else "--"
    drv_status = "OK" if (result["drivable_mask"] is not None and result["drivable_mask"].any()) else "--"
    cv2.putText(frame, f"Lane    : {lane_status}", (25, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, LANE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Drivable: {drv_status}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48, DRIVABLE_COLOR, 1, cv2.LINE_AA)

    closest = result["closest"]
    if closest:
        cv2.putText(frame, f"Closest : {closest['name'].upper()}", (25, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, closest["class_color"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Distance: {closest['distance']:.1f} m", (25, 134),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Risk    : {closest['risk']}", (25, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, closest["risk_color"], 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Closest : None", (25, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Risk    : SAFE", (25, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"BRAKE: {int(result['brake_pct'])}%", (bar_x + bar_w + 12, bar_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, result["brake_col"], 2, cv2.LINE_AA)
    if closest and closest["speed"] > 0:
        cv2.putText(frame, f"{closest['speed']:.0f} km/h", (bar_x - 90, bar_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 80), 2, cv2.LINE_AA)

    # NOTE: the top-right "STRAIGHT / LEFT / RIGHT" direction text badge has
    # been removed from the HUD too. result["direction"] is still computed
    # every frame (see process_frame) and still drives corridor-safety /
    # animal-avoidance / logging -- it's just no longer drawn on screen.

    return frame


# ============================================================
# 9. Event-capture helper (web report gallery) -- shared so both the
#    desktop script and the Flask engine log identical event types.
# ============================================================
def capture_events(result: dict, web_reporter, frame):
    if web_reporter is None:
        return
    closest = result["closest"]
    risk_now = result["risk_now"]
    brake_pct = result["brake_pct"]

    if risk_now == "DANGER" and closest:
        ttc_str = f"{closest['ttc']:.1f}s" if closest["ttc"] != float("inf") else "--"
        web_reporter.capture_event(frame, "DANGER", {
            "dist": f"{closest['distance']:.1f} m", "ttc": ttc_str,
            "object": closest["name"], "kind": closest["kind"],
            "speed": f"{closest['speed']:.1f} km/h",
        })
    elif risk_now == "WARNING" and closest:
        web_reporter.capture_event(frame, "WARNING", {
            "dist": f"{closest['distance']:.1f} m", "object": closest["name"],
            "kind": closest["kind"],
        }, cooldown=3.0)

    if brake_pct > 65 and closest:
        web_reporter.capture_event(frame, "BRAKE", {
            "brake": f"{brake_pct:.0f}%", "dist": f"{closest['distance']:.1f} m",
            "speed": f"{closest['speed']:.1f} km/h",
        })

    if result["potholes"]:
        x1, y1, x2, y2 = result["potholes"][0]
        d, is_est = ground_distance(y2, frame.shape[0], box_height_px=(y2 - y1))
        if d is not None:
            w_cm, h_cm = ground_size_cm(x1, y1, x2, y2, d)
            dist_str, size_str = f"{'~' if is_est else ''}{d:.1f} m", f"{w_cm}x{h_cm} cm"
        else:
            dist_str, size_str = "--", "--"
        web_reporter.capture_event(frame, "POTHOLE", {"dist": dist_str, "size": size_str}, cooldown=5.0)

    if result["humps"]:
        x1, y1, x2, y2 = result["humps"][0]
        d, is_est = ground_distance(y2, frame.shape[0], box_height_px=(y2 - y1))
        if d is not None:
            w_cm, h_cm = ground_size_cm(x1, y1, x2, y2, d)
            dist_str, size_str = f"{'~' if is_est else ''}{d:.1f} m", f"{w_cm}x{h_cm} cm"
        else:
            dist_str, size_str = "--", "--"
        web_reporter.capture_event(frame, "HUMP", {"dist": dist_str, "size": size_str}, cooldown=5.0)

    animals = [o for o in result["obstacles"] if o["kind"] == "animal"]
    if animals:
        closest_animal = min(animals, key=lambda a: a["distance"])
        web_reporter.capture_event(frame, "WARNING", {
            "dist": f"{closest_animal['distance']:.1f} m", "object": closest_animal["name"],
            "kind": "animal",
        }, cooldown=4.0)
