"""
adas_full_pipeline.py  (v3 — animals + trajectory + shared engine)
=================================
Desktop cv2.imshow ADAS app. All the actual model/processing logic now
lives in adas_core.py (shared with the new Flask app in web_app.py /
adas_engine.py) -- this file is just the video loop, window management,
and the 3D window, exactly like before.

New in v3:
  - Animal detection (animal_best.pt) fully wired into the same
    obstacle/risk/brake/direction/trajectory/logging/3D pipeline as
    vehicles.
  - Real corridor-safety direction guidance (direction_predictor v3).
  - trajectory_planner.py road-path overlay as the primary visual
    indicator (small arrow badge kept as a secondary indicator).

Run:
    python adas_full_pipeline.py
"""

import cv2
import os
import time

import adas_core as core
from live_graph import LiveDashboard
from data_logger import DataLogger
from alert_system import AlertSystem
from car_renderer_3d import Scene3D
from render_3d import render_3d_view
from web_report_generator import WebReportGenerator

TESLA_MODE = False

VIDEO_SOURCE = "pothole.mp4"       # >>> change to your file / 0 for webcam <<<
SHOW_3D_WINDOW = True
SAVE_3D_VIDEO = True
THREED_OUTPUT_PATH = "reports/3d_view.mp4"


def main():
    models = core.AdasModels()
    session = core.AdasSession(tesla_mode=TESLA_MODE)

    dashboard = LiveDashboard()
    session.data_logger = DataLogger("logs/session_log.csv")
    session.web_reporter = WebReportGenerator("reports")
    alert = AlertSystem()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 1:
        source_fps = 30.0
    frame_interval = 1.0 / source_fps

    scene_3d = None
    writer_3d = None

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        result = core.process_frame(models, session, frame, current_time)

        if session.frame_count % 30 == 0 or session.frame_count == 1:
            print(f"[frame {session.frame_count}] "
                  f"vehicles={len(result['vehicles'])} animals={len(result['animals'])} "
                  f"potholes={len(result['potholes'])} humps={len(result['humps'])}")

        # ---- 3D view ----
        if SHOW_3D_WINDOW or SAVE_3D_VIDEO:
            if scene_3d is None:
                scene_3d = Scene3D(frame.shape, core.FOCAL_LENGTH)
                if SAVE_3D_VIDEO:
                    out_dir_3d = os.path.dirname(THREED_OUTPUT_PATH)
                    if out_dir_3d:
                        os.makedirs(out_dir_3d, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer_3d = cv2.VideoWriter(THREED_OUTPUT_PATH, fourcc, source_fps,
                                                 (frame.shape[1], frame.shape[0]))

            canvas_3d = render_3d_view(
                result["obstacles"], frame.shape, core.FOCAL_LENGTH, scene_3d,
                drivable_mask=result["drivable_mask"], lane_mask=result["lane_mask"],
                potholes=result["potholes"], humps=result["humps"], signs=result["signs"])
            if SHOW_3D_WINDOW:
                cv2.imshow("3D View - Tesla Style", canvas_3d)
            if writer_3d is not None:
                writer_3d.write(canvas_3d)

        alert.check(result["risk_now"])
        if result["closest"]:
            dashboard.update(result["closest"]["distance"], result["closest"]["speed"],
                              result["closest"]["ttc"], result["closest"]["risk"])

        core.capture_events(result, session.web_reporter, frame)

        frame = core.draw_overlay(frame, result, session)
        frame = alert.draw_flash(frame, result["risk_now"])

        window_title = ("ADAS - Full Pipeline [TESLA MODE]"
                        if TESLA_MODE else "ADAS - Full Pipeline [NORMAL COLOR]")
        cv2.imshow(window_title, frame)

        elapsed_ms = int((time.time() - loop_start) * 1000)
        remaining_ms = max(1, int(frame_interval * 1000) - elapsed_ms)
        if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
            break

    cap.release()
    if writer_3d is not None:
        writer_3d.release()
        print(f"[main] 3D view saved -> {THREED_OUTPUT_PATH}")
    cv2.destroyAllWindows()
    dashboard.close()
    session.data_logger.close()

    session_label = time.strftime("Run %Y-%m-%d %H:%M")
    session.web_reporter.generate(session.data_logger.filepath, session_name=session_label)
    print("[main] Done.")


if __name__ == "__main__":
    main()
