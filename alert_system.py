"""
alert_system.py
------------------
Sound + on-screen alert for DANGER-level collision risk.
Uses winsound (built into Windows, nothing to install) with a cooldown so
it doesn't spam a beep every single frame while risk stays DANGER.

Usage (from test_ai.py):

    from alert_system import AlertSystem
    alert = AlertSystem()
    ...
    risk_now = closest_vehicle["risk"] if closest_vehicle else "SAFE"
    alert.check(risk_now)                          # sound, inside frame loop
    frame = alert.draw_flash(frame, risk_now)       # visual, right before cv2.imshow
"""

import threading
import time

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

import cv2


class AlertSystem:
    def __init__(self, cooldown=1.5, freq=1500, duration=250):
        """
        cooldown : minimum seconds between beeps (avoids audio spam on sustained DANGER)
        freq     : beep frequency in Hz (Windows only)
        duration : beep length in ms
        """
        self.cooldown = cooldown
        self.freq = freq
        self.duration = duration
        self._last_beep = 0.0
        self._flash_state = False
        self._flash_toggle_time = 0.0

    def _beep(self):
        if _HAS_WINSOUND:
            # runs in a thread so winsound.Beep (which blocks) doesn't stall the video loop
            threading.Thread(
                target=lambda: winsound.Beep(self.freq, self.duration), daemon=True
            ).start()
        else:
            print("\a", end="", flush=True)  # terminal bell fallback on non-Windows

    def check(self, risk_label):
        """Call once per frame. Beeps on DANGER, respecting the cooldown."""
        if risk_label == "DANGER":
            now = time.time()
            if now - self._last_beep >= self.cooldown:
                self._beep()
                self._last_beep = now

    def draw_flash(self, frame, risk_label):
        """Optional: draws a pulsing red border + text on the frame during DANGER."""
        if risk_label != "DANGER":
            return frame

        now = time.time()
        if now - self._flash_toggle_time > 0.25:
            self._flash_state = not self._flash_state
            self._flash_toggle_time = now

        if self._flash_state:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12, cv2.LINE_AA)
            cv2.putText(frame, "!! COLLISION WARNING !!", (max(10, w // 2 - 180), 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return frame
