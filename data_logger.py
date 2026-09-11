"""
data_logger.py
----------------
CSV session logger for CollisionAI. Records every detected vehicle's
telemetry on every frame so it can be reviewed later or fed into
report_generator.py.

Usage (from test_ai.py):

    from data_logger import DataLogger
    logger = DataLogger("logs/session_log.csv")
    ...
    logger.log(frame_count, current_time, vehicles)   # inside frame loop
    ...
    logger.close()                                     # after cap.release()
"""

import csv
import os
import time


class DataLogger:
    def __init__(self, filepath="logs/session_log.csv"):
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

        self.filepath = filepath
        self.start_time = time.time()
        self._file = open(filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["frame", "time_s", "kind", "name", "distance_m", "speed_kmh", "ttc_s", "risk"]
        )

    def log(self, frame_count, current_time, vehicles):
        """vehicles: list of obstacle dicts (vehicles AND/OR animals --
        same shape, distinguished by the optional 'kind' key, which
        defaults to 'vehicle' for older callers that don't set it)."""
        t = current_time - self.start_time
        for v in vehicles:
            ttc = v["ttc"]
            ttc_val = "" if ttc == float("inf") else round(ttc, 2)
            self._writer.writerow([
                frame_count,
                round(t, 3),
                v.get("kind", "vehicle"),
                v["name"],
                round(v["distance"], 2),
                round(v["speed"], 2),
                ttc_val,
                v["risk"],
            ])

        # flush periodically rather than every row, so disk I/O doesn't slow the video loop
        if frame_count % 30 == 0:
            self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()
        print(f"[data_logger] Session saved to {self.filepath}")
