// Polls /api/status every 1.5s to keep the dashboard home page's live
// numbers current while a video/camera session is running. No-op on
// pages without a #live-status block.
(function () {
  const box = document.getElementById("live-status");
  if (!box) return;
  const endpoint = box.dataset.poll;

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  async function tick() {
    try {
      const res = await fetch(endpoint);
      const s = await res.json();
      setText("s-status", s.active ? "RUNNING" : "IDLE");
      setText("s-source", s.source || "—");
      setText("s-vehicles", s.active ? s.vehicle_count : 0);
      setText("s-animals", s.active ? s.animal_count : 0);
      setText("s-potholes", s.active ? s.pothole_count : 0);
      setText("s-humps", s.active ? s.hump_count : 0);
      setText("s-closest", s.closest || "—");
      setText("s-distance", s.distance != null ? s.distance + " m" : "—");
      setText("s-speed", s.speed != null ? s.speed + " km/h" : "—");
      setText("s-ttc", s.ttc != null ? s.ttc + " s" : "—");
      setText("s-risk", s.active ? s.risk : "SAFE");
      setText("s-brake", (s.active ? s.brake_pct : 0) + "%");
      setText("s-direction", s.direction || "STRAIGHT");
      setText("s-trajectory", s.trajectory_status || "OK");
    } catch (e) {
      // pipeline idle / server not reachable yet -- ignore and retry
    }
  }

  tick();
  setInterval(tick, 1500);
})();
