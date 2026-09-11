"""
Live/video inference script for ADAS vehicle detection model
(adas_40000_yolov8n - object detection, boxes)

Usage:
    python infer_detect.py --weights best.pt --source video.mp4 --video
    python infer_detect.py --weights best.pt --source 0 --video      # webcam
    python infer_detect.py --weights best.pt --source image.jpg      # single image
"""

import argparse
import os
import time
import cv2
from ultralytics import YOLO


def run_image(model, source, out_dir, imgsz, conf, iou):
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        project=out_dir,
        name="predict",
        exist_ok=True,
    )
    r = results[0]
    print("Class names:", r.names)
    print(f"Saved output under: {os.path.join(out_dir, 'predict')}")


def run_video(model, source, out_dir, imgsz, conf, iou):
    results_gen = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        stream=True,
        save=False,   # we handle drawing + saving ourselves for a live window
    )

    save_dir = os.path.join(out_dir, "video_predict")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "detections.mp4")

    writer = None
    frame_count = 0
    window_name = "ADAS Vehicle Detection - LIVE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    prev_time = time.time()

    for r in results_gen:
        frame = r.plot()  # ultralytics built-in box+label renderer (fine for real detection)

        curr_time = time.time()
        fps_live = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {fps_live:.1f}  Frame: {frame_count}",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
        )

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        writer.write(frame)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped early by user (q pressed).")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}...")

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Done. Processed {frame_count} frames.")
    print(f"Video output saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ADAS vehicle detection inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image, video path, or 0 for webcam")
    parser.add_argument("--out", type=str, default="./infer_out", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--video", action="store_true", help="Set this flag if source is a video/webcam")
    args = parser.parse_args()

    model = YOLO(args.weights)
    print("Class names:", model.names)

    os.makedirs(args.out, exist_ok=True)

    if args.video or str(args.source) == "0":
        run_video(model, args.source, args.out, args.imgsz, args.conf, args.iou)
    else:
        run_image(model, args.source, args.out, args.imgsz, args.conf, args.iou)


if __name__ == "__main__":
    main()
