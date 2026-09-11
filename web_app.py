"""
web_app.py
------------------
Flask web dashboard for CollisionAI/ADAS. Does NOT replace the desktop
app (adas_full_pipeline.py) -- separate entry point, reuses the same
processing engine via adas_engine.py / adas_core.py.

Run:
    python web_app.py
Open:
    http://127.0.0.1:5000
"""

import os
import csv as csvmod
import uuid

from flask import (Flask, render_template, request, redirect, url_for,
                    Response, jsonify, send_file, abort, flash)
from werkzeug.utils import secure_filename

import adas_engine as engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__)
app.secret_key = "adas-dashboard-dev-key"

_camera_session = None  # single shared CameraSession, started/stopped via routes


def _allowed_video(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXT


# ============================================================
# Dashboard home
# ============================================================
@app.route("/")
def index():
    status = engine.live_status.snapshot()
    sessions = engine.list_sessions()[:5]
    return render_template("index.html", status=status, sessions=sessions)


@app.route("/api/status")
def api_status():
    return jsonify(engine.live_status.snapshot())


# ============================================================
# Video page
# ============================================================
@app.route("/video", methods=["GET", "POST"])
def video_page():
    if request.method == "POST":
        f = request.files.get("video_file")
        if not f or f.filename == "":
            flash("Please choose a video file first.")
            return redirect(url_for("video_page"))
        if not _allowed_video(f.filename):
            flash(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXT))}")
            return redirect(url_for("video_page"))
        safe_name = f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}"
        path = os.path.join(UPLOAD_DIR, safe_name)
        f.save(path)
        return redirect(url_for("video_page", path=safe_name))

    selected = request.args.get("path")
    if selected and not os.path.exists(os.path.join(UPLOAD_DIR, selected)):
        selected = None
    return render_template("video.html", selected=selected)


@app.route("/video_feed")
def video_feed():
    filename = request.args.get("path")
    if not filename:
        abort(400, "Missing video path")
    full_path = os.path.join(UPLOAD_DIR, secure_filename(filename))
    if not os.path.exists(full_path):
        abort(404, "Video not found. Please re-upload.")
    return Response(engine.process_video_stream(full_path, source_label=f"Video: {filename}"),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# Camera page
# ============================================================
@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/camera_feed")
def camera_feed():
    global _camera_session
    if _camera_session is None:
        _camera_session = engine.CameraSession(cam_index=0)
    try:
        return Response(_camera_session.frames(),
                         mimetype="multipart/x-mixed-replace; boundary=frame")
    except RuntimeError as e:
        abort(503, str(e))


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    global _camera_session
    if _camera_session is not None:
        _camera_session.stop()
        _camera_session = None
    return jsonify({"stopped": True})


# ============================================================
# Analytics page
# ============================================================
@app.route("/analytics")
def analytics_page():
    sessions = engine.list_sessions()
    session_id = request.args.get("session") or (sessions[0]["id"] if sessions else None)
    return render_template("analytics.html", sessions=sessions, session_id=session_id)


@app.route("/api/session/<session_id>/data")
def api_session_data(session_id):
    meta = engine.get_session_meta(session_id)
    if not meta:
        abort(404)
    csv_path = meta["csv_path"]
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for r in csvmod.DictReader(f):
                rows.append(r)

    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)

    t, dist, speed, ttc, risk, brake_placeholder = [], [], [], [], [], []
    vehicle_ct, animal_ct = [], []
    risk_num = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
    for frame_id in sorted(by_frame, key=lambda x: int(x)):
        group = by_frame[frame_id]
        closest = min(group, key=lambda r: float(r["distance_m"]))
        t.append(float(closest["time_s"]))
        dist.append(float(closest["distance_m"]))
        speed.append(float(closest["speed_kmh"]))
        ttc.append(float(closest["ttc_s"]) if closest["ttc_s"] else 10.0)
        risk.append(risk_num.get(closest["risk"], 0))
        vehicle_ct.append(sum(1 for r in group if r.get("kind", "vehicle") == "vehicle"))
        animal_ct.append(sum(1 for r in group if r.get("kind") == "animal"))

    danger = sum(1 for r in rows if r["risk"] == "DANGER")
    warning = sum(1 for r in rows if r["risk"] == "WARNING")

    return jsonify({
        "time": t, "distance": dist, "speed": speed, "ttc": ttc, "risk": risk,
        "vehicle_count": vehicle_ct, "animal_count": animal_ct,
        "danger_events": danger, "warning_events": warning,
        "meta": meta,
    })


# ============================================================
# Reports pages
# ============================================================
@app.route("/reports")
def reports_page():
    return render_template("reports.html", sessions=engine.list_sessions())


@app.route("/report/<session_id>")
def report_detail(session_id):
    meta = engine.get_session_meta(session_id)
    if not meta:
        abort(404)
    return render_template("report_detail.html", session=meta)


@app.route("/report/<session_id>/view")
def report_view_inline(session_id):
    """Serves report.html inline (not as a download) so it can be embedded
    in an <iframe> on the report_detail page."""
    meta = engine.get_session_meta(session_id)
    if not meta or not os.path.exists(meta["report_path"]):
        abort(404)
    return send_file(meta["report_path"])


@app.route("/download/<session_id>")
def download_report(session_id):
    meta = engine.get_session_meta(session_id)
    if not meta or not os.path.exists(meta["report_path"]):
        abort(404)
    return send_file(meta["report_path"], as_attachment=True,
                      download_name=f"report_{session_id}.html")


@app.route("/download-csv/<session_id>")
def download_csv(session_id):
    meta = engine.get_session_meta(session_id)
    if not meta or not os.path.exists(meta["csv_path"]):
        abort(404)
    return send_file(meta["csv_path"], as_attachment=True,
                      download_name=f"session_{session_id}.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
