"""
web_report_generator.py
---------------------------
Captures key frames during a CollisionAI session and produces a
self-contained report.html after the run.

Includes:
  • Summary stat cards (danger count, min distance, min TTC, potholes,
    humps, brake events, …)
  • Interactive Chart.js graphs: distance, speed, TTC, risk, brake timeline
  • Risk doughnut + per-vehicle breakdown
  • Photo gallery with tabs: DANGER / WARNING / POTHOLE / HUMP / BRAKE
  • Chronological event timeline with metadata

v2 fixes:
  - A real TTC of 0.0s (i.e. an actual imminent collision) used to get
    silently replaced with 10.0 in the chart data because of a
    `cl["ttc_s"] or 10.0` falsy check — 0.0 is falsy in Python, so the
    single most important TTC readings were being erased. Fixed with an
    explicit `is not None` check.
  - Added POTHOLE / HUMP / BRAKE event counts as their own stat cards so
    the summary actually reflects everything that was captured, not just
    DANGER/WARNING vehicle risk.

Usage:
    from web_report_generator import WebReportGenerator

    reporter = WebReportGenerator("reports")
    ...
    # Inside frame loop:
    reporter.capture_event(frame, "DANGER",
        {"dist": "8.3m", "ttc": "1.4s", "vehicle": "car"})
    reporter.capture_event(frame, "POTHOLE",
        {"dist": "12m", "size": "45x30 cm"}, cooldown=5.0)
    ...
    # After cap.release():
    reporter.generate("logs/session_log.csv")
"""

import base64
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np


# -----------------------------------------------------------------------
# Helper: encode a BGR frame to base64 JPEG
# -----------------------------------------------------------------------
def _frame_b64(frame, quality: int = 72) -> str | None:
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


# -----------------------------------------------------------------------
class WebReportGenerator:
    """Captures frames + metadata during a session, then writes report.html."""

    _BADGE_COLORS = {
        "DANGER":  ("#e74c3c", "rgba(231,76,60,.18)"),
        "WARNING": ("#f39c12", "rgba(243,156,18,.18)"),
        "POTHOLE": ("#a55eea", "rgba(165,94,234,.18)"),
        "HUMP":    ("#00bcd4", "rgba(0,188,212,.18)"),
        "BRAKE":   ("#ff6b6b", "rgba(255,107,107,.18)"),
    }

    def __init__(self, out_dir: str = "reports", default_cooldown: float = 2.0):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.events: list[dict] = []
        self._last_cap: dict[str, float] = defaultdict(float)
        self._default_cd = default_cooldown
        self.start_time = time.time()
        self.brake_history: list[tuple[float, float]] = []  # (time_s, brake_pct)

    # ------------------------------------------------------------------
    def capture_event(self, frame, event_type: str,
                      metadata: dict | None = None,
                      cooldown: float | None = None):
        """
        Grab a JPEG snapshot of `frame` and record it with `metadata`.

        event_type : 'DANGER', 'WARNING', 'POTHOLE', 'HUMP', or 'BRAKE'
        cooldown   : override per-type cooldown in seconds
        """
        now = time.time()
        cd  = cooldown if cooldown is not None else self._default_cd
        if now - self._last_cap[event_type] < cd:
            return
        self._last_cap[event_type] = now

        b64 = _frame_b64(frame)
        if b64 is None:
            return

        self.events.append({
            "type":      event_type,
            "time_s":    round(now - self.start_time, 2),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "img_b64":   b64,
            "meta":      metadata or {},
        })

    def log_brake(self, time_s: float, pct: float):
        """Call each frame to track brake history for the brake chart."""
        self.brake_history.append((round(time_s, 2), round(pct, 1)))

    # ------------------------------------------------------------------
    def generate(self, csv_path: str, session_name: str | None = None) -> str | None:
        """Read session CSV + stored events → write reports/report.html."""
        if not os.path.exists(csv_path):
            print(f"[web_report] CSV not found at '{csv_path}' — skipping.")
            return None

        rows = self._load_csv(csv_path)
        if not rows:
            print("[web_report] CSV is empty — nothing to report.")
            return None

        name   = session_name or datetime.now().strftime("Session %Y-%m-%d %H:%M")
        html   = self._build_html(rows, name)
        out_path = os.path.join(self.out_dir, "report.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[web_report] ✅  Report saved → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_csv(path: str) -> list[dict]:
        rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    r["frame"]      = int(r["frame"])
                    r["time_s"]     = float(r["time_s"])
                    r["distance_m"] = float(r["distance_m"])
                    r["speed_kmh"]  = float(r["speed_kmh"])
                    r["ttc_s"]      = float(r["ttc_s"]) if r.get("ttc_s") else None
                    rows.append(r)
                except (ValueError, KeyError):
                    pass
        return rows

    # ---- chart data extraction ----
    def _timeline_data(self, rows: list[dict]):
        by_frame = defaultdict(list)
        for r in rows:
            by_frame[r["frame"]].append(r)

        RISK_NUM = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
        t, dist, speed, ttc, risk = [], [], [], [], []
        for fr in sorted(by_frame):
            cl = min(by_frame[fr], key=lambda x: x["distance_m"])
            t.append(cl["time_s"])
            dist.append(round(cl["distance_m"], 2))
            speed.append(round(cl["speed_kmh"], 2))
            # NOTE: `cl["ttc_s"] or 10.0` used to silently turn a real
            # TTC of 0.0 (imminent collision!) into 10.0 because 0.0 is
            # falsy in Python. Explicit None-check fixes that.
            raw_ttc = cl["ttc_s"] if cl["ttc_s"] is not None else 10.0
            ttc.append(round(min(raw_ttc, 10.0), 2))
            risk.append(RISK_NUM.get(cl.get("risk", "SAFE"), 0))

        bt = [b[0] for b in self.brake_history]
        bv = [b[1] for b in self.brake_history]
        return t, dist, speed, ttc, risk, bt, bv

    def _stats(self, rows: list[dict]) -> dict:
        counts   = defaultdict(int)
        for r in rows:
            counts[r.get("risk", "SAFE")] += 1
        total    = max(sum(counts.values()), 1)
        min_dist = min(r["distance_m"] for r in rows)
        ttc_vals = [r["ttc_s"] for r in rows if r.get("ttc_s") is not None]
        min_ttc  = round(min(ttc_vals), 2) if ttc_vals else None

        by_frame = defaultdict(list)
        for r in rows:
            by_frame[r["frame"]].append(r)

        # duration from max time_s
        max_t = max(r["time_s"] for r in rows)

        # unique vehicles
        all_names = set(r.get("name", "") for r in rows)

        # event-type counts (pothole / hump / brake weren't previously
        # summarised anywhere in the stat cards even though they were
        # captured in the gallery)
        event_counts = defaultdict(int)
        for e in self.events:
            event_counts[e["type"]] += 1

        return {
            "total":       total,
            "frames":      len(by_frame),
            "danger":      counts.get("DANGER", 0),
            "warning":     counts.get("WARNING", 0),
            "safe":        counts.get("SAFE", 0),
            "danger_pct":  round(counts.get("DANGER", 0) / total * 100, 1),
            "warning_pct": round(counts.get("WARNING", 0) / total * 100, 1),
            "min_dist":    round(min_dist, 2),
            "min_ttc":     min_ttc,
            "duration":    round(max_t, 1),
            "events":      len(self.events),
            "vehicles":    ", ".join(sorted(n for n in all_names if n)) or "—",
            "potholes":    event_counts.get("POTHOLE", 0),
            "humps":       event_counts.get("HUMP", 0),
            "brakes":      event_counts.get("BRAKE", 0),
        }

    # ---- HTML building blocks ----
    def _stats_cards_html(self, s: dict) -> str:
        return f"""
<div class="stat-card s-danger">
  <div class="sv">{s['danger']}</div>
  <div class="sl">DANGER Events</div>
  <div class="ss">{s['danger_pct']}% of detections</div>
</div>
<div class="stat-card s-warning">
  <div class="sv">{s['warning']}</div>
  <div class="sl">WARNING Events</div>
  <div class="ss">{s['warning_pct']}% of detections</div>
</div>
<div class="stat-card s-safe">
  <div class="sv">{s['min_dist']} m</div>
  <div class="sl">Closest Approach</div>
  <div class="ss">Minimum distance seen</div>
</div>
<div class="stat-card s-info">
  <div class="sv">{s['min_ttc'] if s['min_ttc'] is not None else '--'} s</div>
  <div class="sl">Minimum TTC</div>
  <div class="ss">Time-to-collision floor</div>
</div>
<div class="stat-card s-info">
  <div class="sv">{s['duration']} s</div>
  <div class="sl">Session Duration</div>
  <div class="ss">{s['frames']} frames analysed</div>
</div>
<div class="stat-card s-info">
  <div class="sv">{s['events']}</div>
  <div class="sl">Captured Snapshots</div>
  <div class="ss">Gallery photos saved</div>
</div>
<div class="stat-card s-pothole">
  <div class="sv">{s['potholes']}</div>
  <div class="sl">Potholes Detected</div>
  <div class="ss">Unique capture events</div>
</div>
<div class="stat-card s-hump">
  <div class="sv">{s['humps']}</div>
  <div class="sl">Humps Detected</div>
  <div class="ss">Unique capture events</div>
</div>
<div class="stat-card s-brake">
  <div class="sv">{s['brakes']}</div>
  <div class="sl">Hard-Brake Events</div>
  <div class="ss">Brake % over 65 threshold</div>
</div>"""

    def _gallery_html(self) -> str:
        if not self.events:
            return '<p class="empty">No event snapshots captured during this session.</p>'

        by_type: dict[str, list] = defaultdict(list)
        for e in self.events:
            by_type[e["type"]].append(e)

        ORDER = ["DANGER", "WARNING", "BRAKE", "POTHOLE", "HUMP"]
        tabs  = ""
        panes = ""
        first = True
        for etype in ORDER:
            evlist = by_type.get(etype, [])
            if not evlist:
                continue
            active = "active" if first else ""
            hidden = "" if first else 'style="display:none"'
            first  = False
            tabs  += (f'<button class="tab-btn {active}" '
                      f'onclick="showTab(this,\'{etype}\')">'
                      f'{etype} <span class="cnt">{len(evlist)}</span></button>\n')
            cards = ""
            for ev in evlist:
                meta = " &nbsp;·&nbsp; ".join(
                    f"<b>{k}</b>: {v}" for k, v in ev["meta"].items() if v is not None)
                bc, bg = self._BADGE_COLORS.get(etype, ("#aaa", "rgba(170,170,170,.15)"))
                cards += f"""
<div class="ev-card">
  <img src="data:image/jpeg;base64,{ev['img_b64']}" loading="lazy" alt="{etype}">
  <div class="ev-body">
    <span class="ev-badge" style="color:{bc};background:{bg};border-color:{bc};">{etype}</span>
    <div class="ev-time">⏱ {ev['timestamp']} &nbsp;(+{ev['time_s']} s into session)</div>
    <div class="ev-meta">{meta}</div>
  </div>
</div>"""
            panes += (f'<div id="pane-{etype}" class="gallery-pane" {hidden}>'
                      f'{cards}</div>')

        return f'<div class="gallery-tabs">{tabs}</div>{panes}'

    def _timeline_html(self) -> str:
        if not self.events:
            return '<p class="empty">No events recorded.</p>'
        items = ""
        for ev in sorted(self.events, key=lambda e: e["time_s"]):
            etype = ev["type"]
            bc = self._BADGE_COLORS.get(etype, ("#aaa", ""))[0]
            meta = " · ".join(f"{k}: {v}" for k, v in ev["meta"].items() if v is not None)
            items += f"""
<div class="tl-item" style="--dot:{bc}">
  <div class="tl-time">{ev['timestamp']} &nbsp;·&nbsp; +{ev['time_s']} s</div>
  <div class="tl-title" style="color:{bc}">{etype}</div>
  <div class="tl-detail">{meta or '—'}</div>
</div>"""
        return items

    # ---- master HTML ----
    def _build_html(self, rows: list[dict], session_name: str) -> str:
        s = self._stats(rows)
        t, dist, speed, ttc, risk, bt, bv = self._timeline_data(rows)
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CollisionAI Report – {session_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{{
  --bg:#0c0c12;--card:#14141e;--border:#252535;
  --accent:#00d4ff;--danger:#e74c3c;--warn:#f39c12;--safe:#2ecc71;
  --pothole:#a55eea;--hump:#00bcd4;--brake:#ff6b6b;
  --text:#dde;--muted:#778;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif}}
/* ── HEADER ── */
header{{
  background:linear-gradient(120deg,#09091a,#181830);
  border-bottom:1px solid var(--border);
  padding:22px 40px;display:flex;align-items:center;gap:14px;
}}
.logo{{width:42px;height:42px;background:var(--accent);border-radius:10px;
       display:flex;align-items:center;justify-content:center;font-size:20px}}
header h1{{font-size:1.45rem;font-weight:700}}
header p{{color:var(--muted);font-size:.82rem;margin-top:2px}}
.hbadge{{margin-left:auto;background:rgba(0,212,255,.1);border:1px solid var(--accent);
         color:var(--accent);padding:4px 14px;border-radius:20px;font-size:.76rem;font-weight:600}}
/* ── LAYOUT ── */
main{{padding:30px 40px;max-width:1400px;margin:0 auto}}
.sec-title{{font-size:1rem;font-weight:700;margin:30px 0 14px;display:flex;align-items:center;gap:10px}}
.sec-title::after{{content:'';flex:1;height:1px;background:var(--border)}}
/* ── STAT CARDS ── */
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-bottom:28px}}
.stat-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;position:relative;overflow:hidden}}
.stat-card::before{{content:'';position:absolute;top:0;left:0;width:3px;height:100%}}
.s-danger::before{{background:var(--danger)}}.s-warning::before{{background:var(--warn)}}
.s-safe::before{{background:var(--safe)}}.s-info::before{{background:var(--accent)}}
.s-pothole::before{{background:var(--pothole)}}.s-hump::before{{background:var(--hump)}}
.s-brake::before{{background:var(--brake)}}
.sv{{font-size:1.9rem;font-weight:700;margin-bottom:3px}}
.s-danger .sv{{color:var(--danger)}}.s-warning .sv{{color:var(--warn)}}
.s-safe .sv{{color:var(--safe)}}.s-info .sv{{color:var(--accent)}}
.s-pothole .sv{{color:var(--pothole)}}.s-hump .sv{{color:var(--hump)}}.s-brake .sv{{color:var(--brake)}}
.sl{{font-size:.82rem;font-weight:600}}.ss{{font-size:.72rem;color:var(--muted);margin-top:3px}}
/* ── CHARTS ── */
.risk-row{{display:grid;grid-template-columns:300px 1fr;gap:16px;margin-bottom:16px}}
.charts-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
@media(max-width:860px){{.risk-row,.charts-2{{grid-template-columns:1fr}}}}
.chart-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}}
.chart-card h3{{font-size:.78rem;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:14px}}
/* ── GALLERY ── */
.gallery-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}}
.tab-btn{{background:var(--card);border:1px solid var(--border);color:var(--muted);
          padding:5px 14px;border-radius:20px;cursor:pointer;font-size:.78rem;transition:all .2s}}
.tab-btn.active,.tab-btn:hover{{background:var(--accent);border-color:var(--accent);color:#000;font-weight:700}}
.cnt{{background:rgba(255,255,255,.15);padding:0 5px;border-radius:8px;font-size:.7rem}}
.gallery-pane{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}}
.ev-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;
          overflow:hidden;transition:transform .18s}}
.ev-card:hover{{transform:translateY(-2px)}}
.ev-card img{{width:100%;height:155px;object-fit:cover;display:block}}
.ev-body{{padding:11px}}
.ev-badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.68rem;
           font-weight:700;margin-bottom:6px;border:1px solid currentColor}}
.ev-time{{font-size:.7rem;color:var(--muted)}}
.ev-meta{{font-size:.73rem;color:var(--text);margin-top:4px}}
/* ── TIMELINE ── */
.timeline{{position:relative;padding-left:22px}}
.timeline::before{{content:'';position:absolute;left:7px;top:0;bottom:0;width:2px;background:var(--border)}}
.tl-item{{position:relative;margin-bottom:14px;padding:11px 14px;
          background:var(--card);border:1px solid var(--border);border-radius:8px}}
.tl-item::before{{content:'';position:absolute;left:-19px;top:15px;
                  width:10px;height:10px;border-radius:50%;background:var(--dot,var(--accent))}}
.tl-time{{font-size:.68rem;color:var(--muted);margin-bottom:3px}}
.tl-title{{font-size:.82rem;font-weight:700}}
.tl-detail{{font-size:.72rem;color:var(--muted);margin-top:2px}}
.empty{{color:var(--muted);padding:24px;text-align:center}}
footer{{text-align:center;padding:30px;color:var(--muted);font-size:.74rem;
        border-top:1px solid var(--border);margin-top:30px}}
</style>
</head>
<body>
<header>
  <div class="logo">🚗</div>
  <div>
    <h1>CollisionAI — Session Report</h1>
    <p>{session_name}</p>
  </div>
  <span class="hbadge">ADAS Analytics</span>
</header>
<main>

<div class="sec-title">📊 Session Summary</div>
<div class="stats-grid">{self._stats_cards_html(s)}</div>

<div class="sec-title">📈 Telemetry Charts</div>

<div class="risk-row">
  <div class="chart-card">
    <h3>Risk Distribution</h3>
    <canvas id="cDonut" height="230"></canvas>
  </div>
  <div class="chart-card">
    <h3>Distance to Closest Vehicle (m)</h3>
    <canvas id="cDist" height="230"></canvas>
  </div>
</div>

<div class="charts-2">
  <div class="chart-card">
    <h3>Relative Speed (km/h)</h3>
    <canvas id="cSpeed" height="200"></canvas>
  </div>
  <div class="chart-card">
    <h3>Time To Collision (s)</h3>
    <canvas id="cTTC" height="200"></canvas>
  </div>
</div>

<div class="chart-card" style="margin-bottom:16px">
  <h3>Risk Level Timeline (0 = SAFE · 1 = WARNING · 2 = DANGER)</h3>
  <canvas id="cRisk" height="100"></canvas>
</div>

<div class="chart-card" style="margin-bottom:28px">
  <h3>Brake % Timeline</h3>
  <canvas id="cBrake" height="100"></canvas>
</div>

<div class="sec-title">📸 Event Gallery</div>
{self._gallery_html()}

<div class="sec-title" style="margin-top:28px">🕐 Event Timeline</div>
<div class="timeline">{self._timeline_html()}</div>

</main>
<footer>
  Generated by CollisionAI Web Reporter &nbsp;·&nbsp; {gen_time}
  &nbsp;·&nbsp; {s['total']} detections · vehicles detected: {s['vehicles']}
</footer>

<script>
const T={json.dumps(t)}, DIST={json.dumps(dist)}, SPD={json.dumps(speed)};
const TTC={json.dumps(ttc)}, RISK={json.dumps(risk)};
const BT={json.dumps(bt)}, BV={json.dumps(bv)};

const BASE={{
  plugins:{{
    legend:{{labels:{{color:'#99a'}} }},
    tooltip:{{backgroundColor:'#1a1a2e',titleColor:'#00d4ff',bodyColor:'#ccc'}}
  }},
  scales:{{
    x:{{ticks:{{color:'#556',maxTicksLimit:10}},grid:{{color:'#1e1e2e'}}}},
    y:{{ticks:{{color:'#556'}},grid:{{color:'#1e1e2e'}}}}
  }}
}};
function merge(a,b){{return Object.assign({{}},a,b)}}

// Donut
const safe_n={s['safe']},warn_n={s['warning']},dan_n={s['danger']};
new Chart(document.getElementById('cDonut'),{{
  type:'doughnut',
  data:{{labels:['SAFE','WARNING','DANGER'],
         datasets:[{{data:[safe_n,warn_n,dan_n],
                     backgroundColor:['#2ecc71','#f39c12','#e74c3c'],borderWidth:0}}]}},
  options:{{...BASE,cutout:'62%',scales:{{}},
            plugins:{{...BASE.plugins,
                      legend:{{position:'bottom',labels:{{color:'#99a',padding:10}}}}}}}}
}});

// Distance
new Chart(document.getElementById('cDist'),{{
  type:'line',
  data:{{labels:T,datasets:[{{label:'Distance (m)',data:DIST,
    borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.07)',
    borderWidth:1.5,pointRadius:0,fill:true,tension:.3}}]}},
  options:{{...BASE,elements:{{point:{{radius:0}}}}}}
}});

// Speed
new Chart(document.getElementById('cSpeed'),{{
  type:'line',
  data:{{labels:T,datasets:[{{label:'Speed (km/h)',data:SPD,
    borderColor:'#ffa502',backgroundColor:'rgba(255,165,2,.07)',
    borderWidth:1.5,pointRadius:0,fill:true,tension:.3}}]}},
  options:{{...BASE,elements:{{point:{{radius:0}}}}}}
}});

// TTC
new Chart(document.getElementById('cTTC'),{{
  type:'line',
  data:{{labels:T,datasets:[
    {{label:'TTC (s)',data:TTC,borderColor:'#a55eea',backgroundColor:'rgba(165,94,234,.07)',
      borderWidth:1.5,pointRadius:0,fill:true,tension:.3}},
    {{label:'Danger < 2s',data:T.map(()=>2),borderColor:'#e74c3c',borderWidth:1,
      borderDash:[4,4],pointRadius:0}},
    {{label:'Warning < 4s',data:T.map(()=>4),borderColor:'#f39c12',borderWidth:1,
      borderDash:[4,4],pointRadius:0}}
  ]}},
  options:{{...BASE,elements:{{point:{{radius:0}}}}}}
}});

// Risk timeline
new Chart(document.getElementById('cRisk'),{{
  type:'line',
  data:{{labels:T,datasets:[{{label:'Risk',data:RISK,
    borderColor:RISK.map(r=>r===2?'#e74c3c':r===1?'#f39c12':'#2ecc71'),
    backgroundColor:'rgba(0,212,255,.04)',borderWidth:2,pointRadius:0,
    fill:false,stepped:'before'}}]}},
  options:{{...BASE,
    elements:{{point:{{radius:0}}}},
    scales:{{
      x:BASE.scales.x,
      y:{{...BASE.scales.y,min:-0.2,max:2.3,
          ticks:{{color:'#556',callback:v=>['SAFE','WARNING','DANGER'][Math.round(v)]||''}}}}
    }}
  }}
}});

// Brake timeline
new Chart(document.getElementById('cBrake'),{{
  type:'line',
  data:{{labels:BT,datasets:[{{label:'Brake %',data:BV,
    borderColor:'#ff6b6b',backgroundColor:'rgba(255,107,107,.10)',
    borderWidth:2,pointRadius:0,fill:true,tension:.25}}]}},
  options:{{...BASE,
    elements:{{point:{{radius:0}}}},
    scales:{{x:BASE.scales.x,y:{{...BASE.scales.y,min:0,max:105}}}}
  }}
}});

// Gallery tab switching
function showTab(btn,type){{
  document.querySelectorAll('.gallery-pane').forEach(p=>p.style.display='none');
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  const p=document.getElementById('pane-'+type);
  if(p)p.style.display='grid';
  btn.classList.add('active');
}}
</script>
</body>
</html>"""
