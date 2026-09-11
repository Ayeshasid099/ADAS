"""
report_generator.py
----------------------
Post-run summary report for CollisionAI. Reads the CSV produced by
data_logger.py and generates polished, presentation-ready charts saved
as PNG files (not just shown on screen) plus a plain-text summary —
meant for dropping straight into your slides.

Usage (standalone, after a run):

    python report_generator.py logs/session_log.csv

Or from code (e.g. called automatically at the end of test_ai.py):

    from report_generator import generate_report
    generate_report("logs/session_log.csv", out_dir="reports")
"""

import sys
import os
import csv
from collections import defaultdict

import matplotlib
# IMPORTANT: do NOT call matplotlib.use() here. This module can be imported
# in the same process as live_graph.py, which needs an interactive GUI
# backend (TkAgg) for its live window. Calling matplotlib.use("Agg") at
# import time used to silently switch the WHOLE process to a headless
# backend, which is why the live graph window never appeared. Instead we
# build figures directly via Figure + FigureCanvasAgg below, which renders
# headless without touching the global backend at all.
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

RISK_ORDER = ["SAFE", "WARNING", "DANGER"]
RISK_COLORS = {"SAFE": "#2ecc71", "WARNING": "#f39c12", "DANGER": "#e74c3c"}


def _load(filepath):
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["frame"] = int(r["frame"])
            r["time_s"] = float(r["time_s"])
            r["distance_m"] = float(r["distance_m"])
            r["speed_kmh"] = float(r["speed_kmh"])
            r["ttc_s"] = float(r["ttc_s"]) if r["ttc_s"] else None
            rows.append(r)
    return rows


def generate_report(filepath, out_dir="reports"):
    if not os.path.exists(filepath):
        print(f"[report_generator] No log file found at '{filepath}' — skipping report.")
        return

    os.makedirs(out_dir, exist_ok=True)
    rows = _load(filepath)
    if not rows:
        print("[report_generator] Log file is empty — nothing to report.")
        return

    # reset rcParams in case live_graph.py's dark_background style is active
    # in this process — report charts should always render on a clean light
    # background, since they're meant for slides.
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # ---- Chart 1: Risk level distribution ----
    counts = defaultdict(int)
    for r in rows:
        counts[r["risk"]] += 1
    total = sum(counts.values())

    fig = Figure(figsize=(6, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    labels = [k for k in RISK_ORDER if k in counts]
    values = [counts[k] for k in labels]
    colors = [RISK_COLORS[k] for k in labels]
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v + total * 0.01, f"{v} ({v/total*100:.1f}%)", ha="center", fontsize=9)
    ax.set_title("Risk Level Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frame count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_distribution.png"), dpi=150)

    # ---- Chart 2: Distance / Speed / TTC timeline for the closest vehicle each frame ----
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)

    frames_sorted = sorted(by_frame)
    t_vals, d_vals, s_vals, ttc_vals = [], [], [], []
    for fr in frames_sorted:
        closest = min(by_frame[fr], key=lambda r: r["distance_m"])
        t_vals.append(closest["time_s"])
        d_vals.append(closest["distance_m"])
        s_vals.append(closest["speed_kmh"])
        ttc_vals.append(closest["ttc_s"] if closest["ttc_s"] is not None else 10)

    fig = Figure(figsize=(9, 8))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    axes[0].plot(t_vals, d_vals, color="#2980b9", lw=1.5)
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title("Closest Vehicle Telemetry Over Session", fontsize=13, fontweight="bold")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_vals, s_vals, color="#d35400", lw=1.5)
    axes[1].set_ylabel("Speed (km/h)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_vals, ttc_vals, color="#8e44ad", lw=1.5)
    axes[2].axhline(2.0, color="#e74c3c", ls="--", lw=1, label="Danger threshold")
    axes[2].axhline(4.0, color="#f39c12", ls="--", lw=1, label="Warning threshold")
    axes[2].set_ylabel("TTC (s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "telemetry_timeline.png"), dpi=150)

    # ---- Text summary ----
    min_dist = min(r["distance_m"] for r in rows)
    ttc_known = [r["ttc_s"] for r in rows if r["ttc_s"] is not None]
    min_ttc = min(ttc_known) if ttc_known else None
    danger_frames = counts.get("DANGER", 0)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("CollisionAI Session Summary\n")
        f.write("============================\n")
        f.write(f"Total logged detections : {total}\n")
        f.write(f"Frames analyzed         : {len(frames_sorted)}\n")
        f.write(f"Minimum distance seen   : {min_dist:.2f} m\n")
        if min_ttc is not None:
            f.write(f"Minimum TTC seen        : {min_ttc:.2f} s\n")
        f.write(f"DANGER detections       : {danger_frames} ({danger_frames/total*100:.1f}%)\n")
        for label in RISK_ORDER:
            if label in counts:
                f.write(f"  {label:8s}: {counts[label]} ({counts[label]/total*100:.1f}%)\n")

    print(f"[report_generator] Report saved to '{out_dir}/':")
    print("  - risk_distribution.png")
    print("  - telemetry_timeline.png")
    print("  - summary.txt")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/session_log.csv"
    generate_report(path)
