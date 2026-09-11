"""
live_graph.py
--------------
Real-time telemetry dashboard for CollisionAI.

Shows Distance, Speed, TTC and Risk Level for the closest tracked vehicle
as a live-updating matplotlib window, running alongside the OpenCV video
window (not a static one-shot plot).

Usage (from test_ai.py):

    from live_graph import LiveDashboard
    dashboard = LiveDashboard()
    ...
    dashboard.update(distance, speed_kmh, ttc, risk_label)   # inside frame loop
    ...
    dashboard.close()                                        # after cap.release()
"""

import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt

RISK_MAP = {"SAFE": 0, "WARNING": 1, "DANGER": 2}

# matplotlib.use() is lazy — it doesn't actually import the backend until a
# figure is created, so wrapping just that call in try/except (as an earlier
# version of this file did) does NOT catch a missing backend. We instead try
# to build a real figure under each candidate backend and see which one
# actually works.
_INTERACTIVE_BACKENDS = ("TkAgg", "Qt5Agg", "QtAgg")


class LiveDashboard:
    def __init__(self, maxlen=150, render_every=2, window_seconds=20):
        """
        maxlen         : how many recent samples are kept in the rolling window
        render_every   : redraw every N update() calls (skip frames -> keeps video loop fast)
        window_seconds : width of the visible time axis, in seconds
        """
        self.render_every = render_every
        self.window_seconds = window_seconds
        self._tick = 0
        self.start_time = time.time()

        self.t = deque(maxlen=maxlen)
        self.dist = deque(maxlen=maxlen)
        self.speed = deque(maxlen=maxlen)
        self.ttc = deque(maxlen=maxlen)
        self.risk = deque(maxlen=maxlen)

        self.enabled = self._activate_interactive_backend()
        if not self.enabled:
            print(
                "[live_graph] WARNING: no interactive matplotlib backend is available "
                "(tried TkAgg, Qt5Agg, QtAgg). The live graph window is DISABLED, but "
                "your detection video will keep running normally.\n"
                "[live_graph] Fix: run  pip install PyQt5   (or reinstall Python with "
                "the 'tcl/tk' option checked so tkinter is included)."
            )
            return

        print(f"[live_graph] using matplotlib backend: {matplotlib.get_backend()}")

        plt.style.use("dark_background")
        self.fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        try:
            self.fig.canvas.manager.set_window_title("CollisionAI - Live Telemetry")
        except Exception:
            pass
        self.fig.patch.set_facecolor("#0d0d0d")

        (self.ax_dist, self.ax_speed), (self.ax_ttc, self.ax_risk) = axes

        self.line_dist, = self.ax_dist.plot([], [], color="#00d4ff", lw=1.8)
        self._style_axis(self.ax_dist, "Distance to Closest Vehicle", "m")

        self.line_speed, = self.ax_speed.plot([], [], color="#ffa502", lw=1.8)
        self._style_axis(self.ax_speed, "Relative Speed", "km/h")

        self.line_ttc, = self.ax_ttc.plot([], [], color="#a55eea", lw=1.8)
        self.ax_ttc.axhline(2.0, color="#e74c3c", ls="--", lw=1, label="Danger < 2s")
        self.ax_ttc.axhline(4.0, color="#f39c12", ls="--", lw=1, label="Warning < 4s")
        self.ax_ttc.legend(loc="upper right", fontsize=7, facecolor="#1a1a1a")
        self._style_axis(self.ax_ttc, "Time To Collision", "s")
        self.ax_ttc.set_ylim(0, 10)

        self.ax_risk.axhspan(-0.5, 0.5, color="#2ecc71", alpha=0.15)
        self.ax_risk.axhspan(0.5, 1.5, color="#f39c12", alpha=0.15)
        self.ax_risk.axhspan(1.5, 2.5, color="#e74c3c", alpha=0.15)
        self.line_risk, = self.ax_risk.step([], [], where="post", color="#ffffff", lw=2)
        self._style_axis(self.ax_risk, "Risk Level", "")
        self.ax_risk.set_ylim(-0.5, 2.5)
        self.ax_risk.set_yticks([0, 1, 2])
        self.ax_risk.set_yticklabels(["SAFE", "WARNING", "DANGER"])

        self.fig.tight_layout()
        plt.show(block=False)

    def _activate_interactive_backend(self):
        """Try each candidate GUI backend by actually building a throwaway
        figure with it. Returns True and leaves the working backend active
        on success; returns False if none of them work."""
        for candidate in _INTERACTIVE_BACKENDS:
            try:
                matplotlib.use(candidate, force=True)
                plt.ion()
                test_fig = plt.figure()
                plt.close(test_fig)
                return True
            except Exception:
                plt.close("all")
                continue
        return False

    def _style_axis(self, ax, title, ylabel):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, color="#cccccc", fontsize=8)
        ax.set_xlabel("Time (s)", color="#cccccc", fontsize=8)
        ax.tick_params(colors="#999999", labelsize=7)
        ax.grid(True, color="#333333", linewidth=0.5)

    def update(self, distance, speed_kmh, ttc, risk_label):
        """Call this once per frame with the closest vehicle's current values."""
        if not self.enabled:
            return

        now = time.time() - self.start_time
        ttc_display = min(ttc, 10.0) if ttc != float("inf") else 10.0

        self.t.append(now)
        self.dist.append(distance)
        self.speed.append(speed_kmh)
        self.ttc.append(ttc_display)
        self.risk.append(RISK_MAP.get(risk_label, 0))

        self._tick += 1
        if self._tick % self.render_every != 0:
            return

        self.line_dist.set_data(self.t, self.dist)
        self.line_speed.set_data(self.t, self.speed)
        self.line_ttc.set_data(self.t, self.ttc)
        self.line_risk.set_data(self.t, self.risk)

        x_lo = max(0, now - self.window_seconds)
        x_hi = max(self.window_seconds, now)

        for ax, ydata in ((self.ax_dist, self.dist), (self.ax_speed, self.speed)):
            ax.set_xlim(x_lo, x_hi)
            if ydata:
                ax.set_ylim(0, max(ydata) * 1.2 + 1)

        self.ax_ttc.set_xlim(x_lo, x_hi)
        self.ax_risk.set_xlim(x_lo, x_hi)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if not self.enabled:
            return
        plt.close(self.fig)
