"""
adas_engine.py
------------------
Flask-facing wrapper around adas_core.py.

Desktop mode (adas_full_pipeline.py) keeps using adas_core directly with
its own cv2.imshow loop. This module instead:

  - Loads the models ONCE (expensive) and reuses them across requests.
  - Runs a video file or live camera through adas_core frame-by-frame
    WITHOUT ever opening a cv2 window (Flask can't have one anyway).
  - Streams annotated frames out as MJPEG for /video and /camera.
  - On completion, finalizes a session: closes the CSV logger, generates
    the existing web_report_generator HTML report, and writes a small
    JSON summary + updates the reports index so the dashboard/reports
    pages can list past sessions without re-parsing every CSV each time.

Nothing here duplicates the model/risk/brake/direction math -- it all
comes from adas_core.process_frame / draw_overlay, exactly like the
desktop app. This is the file the spec calls "adas_engine.py" (section 13).
"""

from __future__ import annotations

import os
import io
import json
import time
import threading
import uuid
from datetime import datetime

import cv2

import adas_core as core
from data_logger import DataLogger
from web_report_generator import WebReportGenerator

LOGS_DIR = "logs"
REPORTS_DIR = "reports"
INDEX_PATH = os.path.join(REPORTS_DIR, "sessions_index.json")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

_models = None
_models_lock = threading.Lock()


def get_models() -> core.AdasModels:
    """Lazily load the (expensive) model bundle once, shared by every
    Flask request/session."""
    global _models
    if _models is None:
        with _models_lock:
            if _models is None:
                _models = core.AdasModels()
    return _models


# ============================================================
# Session index (reports list)
# ============================================================
def _load_index() -> list[dict]:
    if not os.path.exists(INDEX_PATH):
        return []
    try:
        with open(INDEX_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_index(entries: list[dict]):
    with open(INDEX_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def list_sessions() -> list[dict]:
    return sorted(_load_index(), key=lambda e: e.get("created_at", ""), reverse=True)


def get_session_meta(session_id: str) -> dict | None:
    for e in _load_index():
        if e["id"] == session_id:
            return e
    return None


def _upsert_session(entry: dict):
    entries = _load_index()
    entries = [e for e in entries if e["id"] != entry["id"]]
    entries.append(entry)
    _save_index(entries)


# ============================================================
# Live status (for the dashboard home page)
# ============================================================
class LiveStatus:
    """Thread-safe box holding the latest processed-frame result for
    whichever session is currently running, so the dashboard home page
    can show live numbers without hitting the video pipeline itself."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {"active": False}

    def update(self, source_label: str, result: dict):
        with self._lock:
            closest = result.get("closest")
            self._data = {
                "active": True,
                "source": source_label,
                "vehicle_count": len(result["vehicles"]),
                "animal_count": len(result["animals"]),
                "pothole_count": len(result["potholes"]),
                "hump_count": len(result["humps"]),
                "closest": closest["name"] if closest else None,
                "distance": round(closest["distance"], 1) if closest else None,
                "speed": round(closest["speed"], 1) if closest else None,
                "ttc": (round(closest["ttc"], 1) if closest and closest["ttc"] != float("inf") else None),
                "risk": result["risk_now"],
                "brake_pct": round(result["brake_pct"], 1),
                "direction": result["direction"],
                "trajectory_status": "BOTH CORRIDORS UNSAFE" if result.get("corridor_safety") == (False, False) else "OK",
                "updated_at": time.time(),
            }

    def clear(self):
        with self._lock:
            self._data = {"active": False}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)


live_status = LiveStatus()


# ============================================================
# Session finalize (write CSV summary + HTML report + index entry)
# ============================================================
def _summarize_csv(csv_path: str) -> dict:
    import csv as _csv
    stats = {"frames": 0, "vehicles": 0, "animals": 0, "danger": 0, "warning": 0,
              "min_distance": None, "min_ttc": None}
    if not os.path.exists(csv_path):
        return stats
    frames_seen = set()
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            frames_seen.add(row["frame"])
            kind = row.get("kind", "vehicle")
            if kind == "animal":
                stats["animals"] += 1
            else:
                stats["vehicles"] += 1
            if row["risk"] == "DANGER":
                stats["danger"] += 1
            elif row["risk"] == "WARNING":
                stats["warning"] += 1
            try:
                d = float(row["distance_m"])
                stats["min_distance"] = d if stats["min_distance"] is None else min(stats["min_distance"], d)
            except (ValueError, KeyError):
                pass
            if row.get("ttc_s"):
                try:
                    t = float(row["ttc_s"])
                    stats["min_ttc"] = t if stats["min_ttc"] is None else min(stats["min_ttc"], t)
                except ValueError:
                    pass
    stats["frames"] = len(frames_seen)
    return stats


def finalize_session(session_id: str, session: core.AdasSession,
                      data_logger: DataLogger, web_reporter: WebReportGenerator,
                      source_label: str, duration_s: float) -> dict:
    data_logger.close()
    session_name = f"{source_label} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    web_reporter.generate(data_logger.filepath, session_name=session_name)

    csv_stats = _summarize_csv(data_logger.filepath)
    event_counts = {}
    for ev in web_reporter.events:
        event_counts[ev["type"]] = event_counts.get(ev["type"], 0) + 1

    entry = {
        "id": session_id,
        "source": source_label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 1),
        "csv_path": data_logger.filepath,
        "report_path": os.path.join(web_reporter.out_dir, "report.html"),
        "frames": csv_stats["frames"],
        "vehicles": csv_stats["vehicles"],
        "animals": csv_stats["animals"],
        "potholes": event_counts.get("POTHOLE", 0),
        "humps": event_counts.get("HUMP", 0),
        "danger_events": csv_stats["danger"],
        "warning_events": csv_stats["warning"],
        "brake_events": event_counts.get("BRAKE", 0),
        "min_distance": csv_stats["min_distance"],
        "min_ttc": csv_stats["min_ttc"],
    }
    _upsert_session(entry)
    live_status.clear()
    return entry


# ============================================================
# Offline video processing -> MJPEG generator (for /video page)
# ============================================================
def process_video_stream(video_path: str, source_label: str = "Video"):
    """Generator yielding MJPEG-encoded annotated frames while processing
    an uploaded video top to bottom. Finalizes the session (report + CSV
    + index entry) once the video ends."""
    models = get_models()
    session = core.AdasSession(tesla_mode=False)
    session_id = uuid.uuid4().hex[:12]
    csv_path = os.path.join(LOGS_DIR, f"{session_id}.csv")
    report_dir = os.path.join(REPORTS_DIR, session_id)

    session.data_logger = DataLogger(csv_path)
    session.web_reporter = WebReportGenerator(report_dir)

    cap = cv2.VideoCapture(video_path)
    start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = core.process_frame(models, session, frame)
            core.capture_events(result, session.web_reporter, frame)
            live_status.update(source_label, result)
            frame = core.draw_overlay(frame, result, session)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()
        finalize_session(session_id, session, session.data_logger, session.web_reporter,
                          source_label, time.time() - start)


# ============================================================
# Live camera -> MJPEG generator (for /camera page)
# ============================================================
class CameraSession:
    """Wraps a live webcam capture so start/stop can be controlled from
    Flask routes instead of an uncontrolled standalone cv2 window."""

    def __init__(self, cam_index: int = 0):
        self.cam_index = cam_index
        self._running = False
        self._session_id = None

    def frames(self):
        models = get_models()
        session = core.AdasSession(tesla_mode=False)
        self._session_id = uuid.uuid4().hex[:12]
        csv_path = os.path.join(LOGS_DIR, f"{self._session_id}.csv")
        report_dir = os.path.join(REPORTS_DIR, self._session_id)
        session.data_logger = DataLogger(csv_path)
        session.web_reporter = WebReportGenerator(report_dir)

        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            live_status.clear()
            raise RuntimeError("Camera unavailable")

        self._running = True
        start = time.time()
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                result = core.process_frame(models, session, frame)
                core.capture_events(result, session.web_reporter, frame)
                live_status.update("Live Camera", result)
                frame = core.draw_overlay(frame, result, session)

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        finally:
            cap.release()
            finalize_session(self._session_id, session, session.data_logger,
                              session.web_reporter, "Live Camera", time.time() - start)

    def stop(self):
        self._running = False
