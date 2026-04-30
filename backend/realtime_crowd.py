"""
Realtime Crowd Monitoring — Continuous RTSP camera monitoring with YOLO head detection
and Gemini-based crowd analysis.

Uses the same camera feeds as Realtime Forensics module.
Captures frames, runs best_head.pt detection, segments into 10-minute windows,
uses Gemini 2.5 Flash for crowd intelligence, logs to CSV, and generates PDF reports.
8-hour max runtime with automatic master report generation.
"""

import csv
import concurrent.futures
import json
import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Re-use forensics infrastructure for Gemini
import forensics as _forensics_mod
from forensics import _ensure_genai, FORENSICS_JPEG_QUALITY, FORENSICS_MAX_OUTPUT_TOKENS

# Import crowd_report helpers for PDF generation
import crowd_report as _cr

# CSRNet density model for heatmaps
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from csrnet.csrnet_heatmap import load_density_model, generate_density_heatmap, overlay_heatmap
_csrnet_loaded = False

# ── Timezone ──
IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(IST)

# ── Config ──
SEGMENT_DURATION_SEC = 10 * 60  # 10 minutes
MAX_RUNTIME_SEC = 8 * 60 * 60   # 8 hours
CAPTURE_INTERVAL = max(3, int(os.getenv("RT_CROWD_CAPTURE_INTERVAL", "5")))  # seconds between captures
BATCH_SIZE = max(4, int(os.getenv("RT_CROWD_BATCH_SIZE", "9")))  # frames per Gemini batch

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
MODEL_PATH = str(MODELS_DIR / "best_head.pt")

DATA_DIR = BACKEND_DIR / "data" / "realtime_crowd"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = DATA_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
HOURLY_DIR = DATA_DIR / "hourly"
HOURLY_DIR.mkdir(parents=True, exist_ok=True)

# Load cameras dynamically from rtsp_links.yml
def _load_cameras_from_config():
    import yaml
    from urllib.parse import urlparse
    config_path = BACKEND_DIR / "config" / "rtsp_links.yml"
    cameras = {}
    urls = {}
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        # Prefer crowd_cameras section (named entries) over rtsp_links
        crowd_cams = cfg.get("crowd_cameras")
        if crowd_cams is not None and isinstance(crowd_cams, list):
            for entry in crowd_cams:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", "").strip()
                url = entry.get("url", "").strip()
                if not name or not url:
                    continue
                cam_id = "cam_" + name
                cameras[cam_id] = name
                urls[cam_id] = url
        else:
            # Fallback: derive from rtsp_links (IP-based names)
            for url in cfg.get("rtsp_links", []):
                host = urlparse(url).hostname or ""
                if not host:
                    continue
                cam_id = "cam_" + host.replace(".", "_")
                cameras[cam_id] = host
                urls[cam_id] = url
    except Exception as e:
        print(f"[RT-CROWD] Failed to load cameras from config: {e}")
    return cameras, urls

CAMERAS, _CAMERA_URLS = _load_cameras_from_config()


def auto_start_if_cameras():
    """Auto-start crowd monitoring if cameras are configured. Called after server startup."""
    if CAMERAS:
        print(f"[RT-CROWD] Auto-starting with {len(CAMERAS)} configured cameras")
        start_session()


# ── Global state ──
_lock = threading.Lock()
_session: Optional[dict] = None
_monitor_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_yolo_model = None
_yolo_lock = threading.Lock()

# Per-camera live analysis store (frame + metrics for dashboard & MJPEG)
_live_cam_lock = threading.Lock()
_live_cam_analysis: Dict[str, dict] = {}
_live_cam_heatmaps: Dict[str, np.ndarray] = {}
_live_cam_counts_history: Dict[str, list] = {}  # rolling window per camera
_live_cam_timestamps: Dict[str, list] = {}
_HEATMAP_DECAY = 0.92
_JPEG_Q = [cv2.IMWRITE_JPEG_QUALITY, 70]
_MAX_HISTORY = 120  # keep last 120 captures (~10 min at 5s intervals)

# Per-camera failure tracking — skip cameras that fail repeatedly
_cam_fail_counts: Dict[str, int] = {}
_cam_fail_lock = threading.Lock()
_CAM_FAIL_THRESHOLD = 3          # consecutive failures before backoff
_CAM_BACKOFF_CYCLES = 6          # skip this many cycles before retrying
_cam_backoff_remaining: Dict[str, int] = {}


def get_live_cam_frame(camera_id: str) -> Optional[bytes]:
    """Return the latest JPEG-encoded annotated frame for a camera."""
    with _live_cam_lock:
        entry = _live_cam_analysis.get(camera_id)
        return entry.get("frame_jpeg") if entry else None


def get_live_cam_analysis_data(camera_id: str) -> Optional[dict]:
    """Return per-camera analysis dict (without raw frame bytes) for JSON API."""
    with _live_cam_lock:
        entry = _live_cam_analysis.get(camera_id)
        if not entry:
            return None
        # Return everything except the raw JPEG bytes
        return {k: v for k, v in entry.items() if k != "frame_jpeg"}


def get_all_cam_analysis() -> dict:
    """Return analysis for all cameras (without frame bytes)."""
    with _live_cam_lock:
        result = {}
        for cid, entry in _live_cam_analysis.items():
            result[cid] = {k: v for k, v in entry.items() if k != "frame_jpeg"}
        return result


def _publish_cam_analysis(camera_id: str, frame: np.ndarray, boxes: list,
                          count: int, camera_name: str):
    """Build heatmap overlay, compute analysis metrics, store for dashboard & MJPEG."""
    h, w = frame.shape[:2]
    ts_str = _now_ist().strftime("%H:%M:%S")

    # ── Update rolling history ──
    with _live_cam_lock:
        if camera_id not in _live_cam_counts_history:
            _live_cam_counts_history[camera_id] = []
            _live_cam_timestamps[camera_id] = []
        hist = _live_cam_counts_history[camera_id]
        ts_hist = _live_cam_timestamps[camera_id]
        hist.append(count)
        ts_hist.append(ts_str)
        if len(hist) > _MAX_HISTORY:
            hist[:] = hist[-_MAX_HISTORY:]
            ts_hist[:] = ts_hist[-_MAX_HISTORY:]

    # ── CSRNet density heatmap ──
    global _csrnet_loaded
    if not _csrnet_loaded:
        _csrnet_loaded = load_density_model()

    try:
        heatmap_bgr, density_map, density_count = generate_density_heatmap(frame)
    except Exception as _hm_err:
        heatmap_bgr, density_map, density_count = None, None, 0

    # Always overlay heatmap — same as irisv3 crowd-worker: addWeighted 50/50
    if heatmap_bgr is not None:
        annotated = overlay_heatmap(frame, heatmap_bgr, alpha=0.5)
    else:
        annotated = frame.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # HUD bar
    bar_h = 32
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    hud = f"IRIS CROWD | {camera_name} | {count} Persons | {ts_str}"
    cv2.putText(annotated, hud, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

    ret, buf = cv2.imencode('.jpg', annotated, _JPEG_Q)
    frame_jpeg = buf.tobytes() if ret else b''

    # ── Compute analysis metrics (rule-based, no Gemini) ──
    cong = _cr.congestion_score(count)
    cond = _cr.condition_from_score(cong)
    fl = _cr.flow_direction(hist) if len(hist) >= 4 else "Stable"
    predicted = _cr.predict_next_window(hist) if len(hist) >= 4 else count
    pred_cond = _cr.condition_from_score(_cr.congestion_score(predicted))
    insights = _cr.insights_from_segment(hist[-30:] if hist else [count], cond, fl, predicted)
    alert_text, alert_level = _cr.safety_alerts(cond, pred_cond)
    free = _cr.free_space_pct(count)

    avg_count = float(np.mean(hist[-30:])) if hist else float(count)
    peak_count = max(hist[-30:]) if hist else count

    analysis_entry = {
        "frame_jpeg": frame_jpeg,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "timestamp": ts_str,
        "count": count,
        "avg_count": round(avg_count, 1),
        "peak_count": peak_count,
        "congestion_score": cong,
        "condition": cond,
        "flow": fl,
        "free_space": free,
        "predicted_count": predicted,
        "predicted_condition": pred_cond,
        "insights": insights,
        "safety_alert": alert_text,
        "safety_level": alert_level,
        "visibility": 90,
        "behavior": "Normal crowd flow" if count < 50 else "Dense crowd activity",
        "sentiment": "NEUTRAL",
        "counts_history": list(hist[-60:]),
        "timestamps_history": list(ts_hist[-60:]),
    }

    with _live_cam_lock:
        _live_cam_analysis[camera_id] = analysis_entry


def clear_live_cam_frames():
    """Clear all live frame/analysis buffers when session ends."""
    with _live_cam_lock:
        _live_cam_analysis.clear()
        _live_cam_heatmaps.clear()
        _live_cam_counts_history.clear()
        _live_cam_timestamps.clear()
    with _cam_fail_lock:
        _cam_fail_counts.clear()
        _cam_backoff_remaining.clear()


def _get_rtsp_url(camera_id: str) -> str:
    return _CAMERA_URLS.get(camera_id, "")


def _load_yolo():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _yolo_model = YOLO(MODEL_PATH, task="detect")
        if device == "cuda":
            _yolo_model.to(device)
        print(f"[RT-CROWD] YOLO model loaded on {device}")
        return _yolo_model


def _capture_frame(camera_id: str) -> Optional[np.ndarray]:
    """Capture a single frame from RTSP camera via FFmpeg.
    Tracks consecutive failures per camera and backs off unreachable ones."""
    # Check backoff — skip cameras that failed repeatedly
    with _cam_fail_lock:
        remaining = _cam_backoff_remaining.get(camera_id, 0)
        if remaining > 0:
            _cam_backoff_remaining[camera_id] = remaining - 1
            return None

    url = _get_rtsp_url(camera_id)
    tmp_path = FRAMES_DIR / f"_tmp_{camera_id[-8:]}_{threading.get_ident()}.jpg"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", "tcp",
                "-timeout", "4000000",
                "-i", url,
                "-ss", "1.5",
                "-frames:v", "1",
                "-q:v", "2",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=6,
        )
        if tmp_path.exists() and tmp_path.stat().st_size > 5000:
            frame = cv2.imread(str(tmp_path))
            tmp_path.unlink(missing_ok=True)
            # Reset failure counter on success
            with _cam_fail_lock:
                _cam_fail_counts[camera_id] = 0
            return frame
        tmp_path.unlink(missing_ok=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)

    # Record failure and start backoff if threshold reached
    with _cam_fail_lock:
        _cam_fail_counts[camera_id] = _cam_fail_counts.get(camera_id, 0) + 1
        if _cam_fail_counts[camera_id] >= _CAM_FAIL_THRESHOLD:
            _cam_backoff_remaining[camera_id] = _CAM_BACKOFF_CYCLES
            print(f"[RT-CROWD] Camera {camera_id} failed {_cam_fail_counts[camera_id]}x consecutively, "
                  f"backing off for {_CAM_BACKOFF_CYCLES} cycles")
    return None


def _detect_heads(model, frame: np.ndarray) -> tuple:
    """Run YOLO head detection, return (boxes, count)."""
    results = model(frame, conf=0.12, iou=0.5, verbose=False, imgsz=480)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if r.names[cls] in ['person', 'people', 'head']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes, len(boxes)


def _gemini_crowd_analysis(frames_data: list, segment_stats: dict, model) -> dict:
    """
    Send batch frames + segment stats to Gemini 2.5 Flash for crowd intelligence.
    Returns dict with all requested analysis fields.
    """
    now_str = _now_ist().strftime("%H:%M:%S")

    camera_descriptions = []
    for fd in frames_data:
        camera_descriptions.append(
            f"Camera '{fd['camera_name']}': {fd['head_count']} persons detected at {fd['capture_time']}"
        )

    prompt_parts = [
        f"""You are an expert crowd safety and surveillance analyst monitoring LIVE camera feeds in real-time.
Current time: {now_str} IST.

You are reviewing frames from {len(frames_data)} cameras for a 10-minute crowd monitoring segment.

Camera observations:
{chr(10).join(camera_descriptions)}

Segment statistics:
- Total persons detected across all cameras: {segment_stats['total_count']}
- Average persons per capture: {segment_stats['avg_count']:.1f}
- Peak persons in single capture: {segment_stats['peak_count']}
- Segment duration: {segment_stats['segment_duration_min']:.1f} minutes
- Number of captures in this segment: {segment_stats['num_captures']}

Analyze the frames and provide a JSON response with EXACTLY these fields:

{{
  "crowd_movement": "A 1-2 sentence description of crowd movement patterns. Examples: 'Crowd movement is stable with no sudden flows', 'Crowd is building up near the main gate', 'Dispersal pattern detected'",
  "crowd_density": "LOW|MODERATE|HIGH|CRITICAL - with a brief explanation",
  "sentiment": "NEUTRAL|CAUTIOUS|AGITATED|MOB - Analyze whether the crowd appears as a mob or neutral. Describe body language and grouping patterns",
  "weapon_detected": "YES|NO - State clearly if anyone appears to be carrying a weapon. Describe if yes",
  "fight_collision_injury": "YES|NO - State if anyone is involved in a fight, collision, or injury. Describe if yes",
  "wrongful_activity": "YES|NO - State if any wrongful or criminal activity is observed. Describe if yes",
  "visibility_score": 0-100,
  "predicted_count_next_segment": <integer>,
  "safety_precaution": "A specific safety precaution step recommended based on current conditions",
  "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
  "per_camera_summary": [
    {{"camera_name": "name", "count": N, "observation": "brief note"}}
  ]
}}

Be factual. Only report what is CLEARLY visible. If uncertain, err on the side of caution.
NEVER report camera quality issues. Focus only on people, crowd behavior, and safety.

Analyze these frames now:
"""
    ]

    for fd in frames_data:
        if fd.get('frame_jpeg'):
            prompt_parts.append({"mime_type": "image/jpeg", "data": fd['frame_jpeg']})
            prompt_parts.append(f"[Camera: {fd['camera_name']}]")

    try:
        response = None
        for attempt in range(4):
            try:
                response = model.generate_content(prompt_parts)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = (attempt + 1) * 15
                    print(f"[RT-CROWD] Rate limited, waiting {wait}s")
                    _stop_event.wait(wait)
                    if _stop_event.is_set():
                        return _fallback_analysis(segment_stats)
                    continue
                raise

        if response is None:
            return _fallback_analysis(segment_stats)

        answer = response.text.strip()
        # Strip markdown
        if answer.startswith("```json"):
            answer = answer[7:]
        if answer.startswith("```"):
            answer = answer[3:]
        if answer.endswith("```"):
            answer = answer[:-3]
        answer = answer.strip()

        left = answer.find("{")
        right = answer.rfind("}")
        if left != -1 and right != -1:
            answer = answer[left:right + 1]

        data = json.loads(answer)
        # Normalize fields
        data.setdefault("crowd_movement", "Stable crowd movement observed.")
        data.setdefault("crowd_density", "MODERATE")
        data.setdefault("sentiment", "NEUTRAL")
        data.setdefault("weapon_detected", "NO")
        data.setdefault("fight_collision_injury", "NO")
        data.setdefault("wrongful_activity", "NO")
        data.setdefault("visibility_score", 90)
        data.setdefault("predicted_count_next_segment", segment_stats['avg_count'])
        data.setdefault("safety_precaution", "Continue standard monitoring.")
        data.setdefault("overall_risk", "LOW")
        data.setdefault("per_camera_summary", [])
        return data

    except Exception as e:
        print(f"[RT-CROWD] Gemini analysis error: {e}")
        return _fallback_analysis(segment_stats)


def _fallback_analysis(stats: dict) -> dict:
    """Rule-based fallback when Gemini is unavailable."""
    avg = stats.get('avg_count', 0)
    peak = stats.get('peak_count', 0)

    if peak > 60:
        density = "HIGH"
        risk = "HIGH"
        movement = "Elevated crowd density detected. Movement patterns indicate congestion."
        precaution = "Deploy additional personnel to manage crowd flow. Monitor for bottlenecks."
    elif peak > 30:
        density = "MODERATE"
        risk = "MEDIUM"
        movement = "Moderate crowd presence. Movement is steady with no sudden changes."
        precaution = "Maintain active monitoring. Prepare crowd management protocols."
    else:
        density = "LOW"
        risk = "LOW"
        movement = "Crowd movement is stable with no sudden flows."
        precaution = "Continue standard monitoring procedures."

    return {
        "crowd_movement": movement,
        "crowd_density": density,
        "sentiment": "NEUTRAL",
        "weapon_detected": "NO",
        "fight_collision_injury": "NO",
        "wrongful_activity": "NO",
        "visibility_score": 90,
        "predicted_count_next_segment": int(avg * 1.05),
        "safety_precaution": precaution,
        "overall_risk": risk,
        "per_camera_summary": [],
    }


# ── CSV Logging ──

def _csv_path() -> Path:
    today = _now_ist().strftime("%Y-%m-%d")
    return DATA_DIR / f"crowd_events_{today}.csv"


def _init_csv():
    """Initialize CSV file with headers if it doesn't exist."""
    path = _csv_path()
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "segment_index", "segment_start", "segment_end",
                "total_count", "avg_count", "peak_count",
                "crowd_movement", "crowd_density", "sentiment",
                "weapon_detected", "fight_collision_injury", "wrongful_activity",
                "visibility_score", "predicted_next", "safety_precaution",
                "overall_risk", "cameras_active",
            ])
    return path


def _log_to_csv(segment_idx: int, seg_start: str, seg_end: str,
                stats: dict, analysis: dict, cameras_active: int):
    path = _init_csv()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            _now_ist().isoformat(),
            segment_idx,
            seg_start,
            seg_end,
            stats.get('total_count', 0),
            f"{stats.get('avg_count', 0):.1f}",
            stats.get('peak_count', 0),
            analysis.get('crowd_movement', ''),
            analysis.get('crowd_density', ''),
            analysis.get('sentiment', ''),
            analysis.get('weapon_detected', ''),
            analysis.get('fight_collision_injury', ''),
            analysis.get('wrongful_activity', ''),
            analysis.get('visibility_score', ''),
            analysis.get('predicted_count_next_segment', ''),
            analysis.get('safety_precaution', ''),
            analysis.get('overall_risk', ''),
            cameras_active,
        ])


# ── Session Management ──

def get_status() -> dict:
    with _lock:
        if _session is None:
            return {
                "active": False,
                "status": "idle",
                "message": "Live crowd monitoring not started.",
                "segments_completed": 0,
                "current_segment": 0,
                "total_captures": 0,
                "events": [],
                "recent_analysis": None,
                "cameras": len(CAMERAS),
                "runtime_sec": 0,
                "max_runtime_sec": MAX_RUNTIME_SEC,
            }
        elapsed = time.time() - _session.get("started_ts", time.time())
        return {
            "active": _session["status"] == "running",
            "status": _session["status"],
            "message": _session.get("message", ""),
            "segments_completed": _session.get("segments_completed", 0),
            "current_segment": _session.get("current_segment", 0),
            "total_captures": _session.get("total_captures", 0),
            "events": _session.get("events", [])[-50:],
            "recent_analysis": _session.get("recent_analysis"),
            "cameras": len(CAMERAS),
            "runtime_sec": int(elapsed),
            "max_runtime_sec": MAX_RUNTIME_SEC,
            "started_at": _session.get("started_at", ""),
            "csv_path": str(_csv_path()) if _csv_path().exists() else None,
            "per_camera_counts": _session.get("per_camera_counts", {}),
            "last_total_count": _session.get("last_total_count", 0),
            "selected_cameras": _session.get("selected_cameras", []),
        }


def add_camera(name: str, url: str) -> dict:
    """Add a camera to crowd_cameras in config and reload. Auto-starts session if not running."""
    global CAMERAS, _CAMERA_URLS
    import yaml
    name = name.strip()
    url = url.strip()
    if not name or not url:
        return {"status": "error", "message": "Name and URL are required."}

    cam_id = "cam_" + name
    if cam_id in CAMERAS:
        return {"status": "error", "message": f"Camera '{name}' already exists."}

    config_path = BACKEND_DIR / "config" / "rtsp_links.yml"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        crowd_cams = cfg.get("crowd_cameras", [])
        if not isinstance(crowd_cams, list):
            crowd_cams = []
        crowd_cams.append({"name": name, "url": url})
        cfg["crowd_cameras"] = crowd_cams
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        return {"status": "error", "message": f"Failed to save config: {e}"}

    # Reload in-memory
    CAMERAS[cam_id] = name
    _CAMERA_URLS[cam_id] = url
    print(f"[RT-CROWD] Camera added: {name} ({cam_id})")

    # Auto-start if not running, or add to running session
    with _lock:
        running = _session and _session["status"] == "running"
        if running and cam_id not in _session.get("selected_cameras", []):
            _session["selected_cameras"].append(cam_id)
    if not running:
        start_session()

    return {"status": "ok", "camera_id": cam_id, "name": name, "cameras": get_available_cameras()}


def remove_camera(name: str) -> dict:
    """Remove a camera from crowd_cameras in config and reload."""
    global CAMERAS, _CAMERA_URLS
    import yaml
    name = name.strip()
    cam_id = "cam_" + name
    if cam_id not in CAMERAS:
        return {"status": "error", "message": f"Camera '{name}' not found."}

    config_path = BACKEND_DIR / "config" / "rtsp_links.yml"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        crowd_cams = cfg.get("crowd_cameras", [])
        if isinstance(crowd_cams, list):
            crowd_cams = [c for c in crowd_cams if isinstance(c, dict) and c.get("name", "").strip() != name]
        cfg["crowd_cameras"] = crowd_cams
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        return {"status": "error", "message": f"Failed to save config: {e}"}

    CAMERAS.pop(cam_id, None)
    _CAMERA_URLS.pop(cam_id, None)
    # Remove from running session
    with _lock:
        if _session and cam_id in _session.get("selected_cameras", []):
            _session["selected_cameras"].remove(cam_id)
    # Clean up live state
    with _live_cam_lock:
        _live_cam_analysis.pop(cam_id, None)
        _live_cam_heatmaps.pop(cam_id, None)
        _live_cam_counts_history.pop(cam_id, None)
        _live_cam_timestamps.pop(cam_id, None)
    print(f"[RT-CROWD] Camera removed: {name} ({cam_id})")
    return {"status": "ok", "name": name, "cameras": get_available_cameras()}


# ── Camera Group Management ──

def _load_config():
    import yaml
    config_path = BACKEND_DIR / "config" / "rtsp_links.yml"
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _save_config(cfg):
    import yaml
    config_path = BACKEND_DIR / "config" / "rtsp_links.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def get_camera_groups() -> list:
    """Return all camera groups."""
    try:
        cfg = _load_config()
        groups = cfg.get("crowd_camera_groups", [])
        if not isinstance(groups, list):
            return []
        return [{"name": g["name"], "cameras": g.get("cameras", []),
                 "count": len(g.get("cameras", []))} for g in groups if isinstance(g, dict) and g.get("name")]
    except Exception:
        return []


def create_camera_group(group_name: str, camera_names: list = None) -> dict:
    """Create a new camera group, optionally with cameras from the current active set."""
    group_name = group_name.strip()
    if not group_name:
        return {"status": "error", "message": "Group name is required."}
    try:
        cfg = _load_config()
        groups = cfg.get("crowd_camera_groups", [])
        if not isinstance(groups, list):
            groups = []
        # Check duplicate
        for g in groups:
            if isinstance(g, dict) and g.get("name", "").strip().lower() == group_name.lower():
                return {"status": "error", "message": f"Group '{group_name}' already exists."}

        # Build camera list
        cams = []
        if camera_names:
            # Use specified cameras
            for cn in camera_names:
                cam_id = "cam_" + cn.strip()
                if cam_id in _CAMERA_URLS:
                    cams.append({"name": cn.strip(), "url": _CAMERA_URLS[cam_id]})
        else:
            # Use all currently active cameras
            for cam_id, name in CAMERAS.items():
                if cam_id in _CAMERA_URLS:
                    cams.append({"name": name, "url": _CAMERA_URLS[cam_id]})

        groups.append({"name": group_name, "cameras": cams})
        cfg["crowd_camera_groups"] = groups
        _save_config(cfg)
        print(f"[RT-CROWD] Group created: '{group_name}' with {len(cams)} cameras")
        return {"status": "ok", "name": group_name, "count": len(cams), "groups": get_camera_groups()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_camera_group(group_name: str) -> dict:
    """Delete a camera group."""
    group_name = group_name.strip()
    try:
        cfg = _load_config()
        groups = cfg.get("crowd_camera_groups", [])
        if not isinstance(groups, list):
            return {"status": "error", "message": "No groups found."}
        new_groups = [g for g in groups if not (isinstance(g, dict) and g.get("name", "").strip().lower() == group_name.lower())]
        if len(new_groups) == len(groups):
            return {"status": "error", "message": f"Group '{group_name}' not found."}
        cfg["crowd_camera_groups"] = new_groups
        _save_config(cfg)
        print(f"[RT-CROWD] Group deleted: '{group_name}'")
        return {"status": "ok", "name": group_name, "groups": get_camera_groups()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def activate_camera_group(group_name: str) -> dict:
    """Stop current session, load cameras from group, and start analysis."""
    global CAMERAS, _CAMERA_URLS
    group_name = group_name.strip()
    try:
        cfg = _load_config()
        groups = cfg.get("crowd_camera_groups", [])
        group = None
        for g in groups:
            if isinstance(g, dict) and g.get("name", "").strip().lower() == group_name.lower():
                group = g
                break
        if not group:
            return {"status": "error", "message": f"Group '{group_name}' not found."}

        group_cams = group.get("cameras", [])
        if not group_cams:
            return {"status": "error", "message": f"Group '{group_name}' has no cameras."}

        # Stop existing session
        _stop_event.set()
        with _lock:
            if _session:
                _session["status"] = "stopped"

        # Clear current cameras and live state
        CAMERAS.clear()
        _CAMERA_URLS.clear()
        with _live_cam_lock:
            _live_cam_analysis.clear()
            _live_cam_heatmaps.clear()
            _live_cam_counts_history.clear()
            _live_cam_timestamps.clear()

        # Load group cameras
        for entry in group_cams:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "").strip()
            url = entry.get("url", "").strip()
            if name and url:
                cam_id = "cam_" + name
                CAMERAS[cam_id] = name
                _CAMERA_URLS[cam_id] = url

        # Update crowd_cameras in config to match the group
        cfg["crowd_cameras"] = list(group_cams)
        _save_config(cfg)

        # Wait a moment for old session to stop, then start new
        import time
        time.sleep(1)
        result = start_session()
        print(f"[RT-CROWD] Activated group '{group_name}' with {len(CAMERAS)} cameras")
        return {"status": "ok", "name": group_name, "cameras": get_available_cameras(),
                "session": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_all_events() -> list:
    """Return all segment events from CSV + in-memory."""
    events = []
    # Read from CSV
    path = _csv_path()
    if path.exists():
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    events.append(dict(row))
        except Exception:
            pass
    return events


def get_available_cameras() -> list:
    """Return list of available cameras with id and name."""
    return [{"id": cid, "name": name} for cid, name in CAMERAS.items()]


def start_session(selected_cameras: list = None) -> dict:
    global _session, _monitor_thread, _stop_event

    with _lock:
        if _session and _session["status"] == "running":
            return {"status": "already_running", "message": "Live crowd monitoring is already active.",
                    "selected_cameras": _session.get("selected_cameras", [])}

    # Validate selected cameras
    if selected_cameras:
        valid = [c for c in selected_cameras if c in CAMERAS]
    else:
        valid = list(CAMERAS.keys())

    if not valid:
        return {"status": "error", "message": "No valid cameras selected."}

    _stop_event = threading.Event()

    with _lock:
        _session = {
            "status": "running",
            "message": "Initializing live crowd monitoring...",
            "started_at": _now_ist().isoformat(),
            "started_ts": time.time(),
            "segments_completed": 0,
            "current_segment": 1,
            "total_captures": 0,
            "events": [],
            "recent_analysis": None,
            "segments_data": [],
            "selected_cameras": valid,
        }

    _monitor_thread = threading.Thread(target=_monitor_loop, daemon=True, name="rt-crowd-monitor")
    _monitor_thread.start()
    cam_names = [CAMERAS[c] for c in valid]
    print(f"[RT-CROWD] Live crowd monitoring started on {len(valid)} cameras: {cam_names}")
    return {"status": "started", "message": f"Monitoring {len(valid)} cameras.",
            "selected_cameras": valid}


def stop_session() -> dict:
    _stop_event.set()
    with _lock:
        if _session is not None:
            _session["status"] = "stopped"
            _session["message"] = "Session stopped."
    print("[RT-CROWD] Session stop requested")
    return {"status": "stopped"}


# ── PDF Report Generation ──

def generate_report_pdf() -> Optional[str]:
    """Generate an on-demand PDF report from all accumulated segment data."""
    with _lock:
        if _session is None:
            return None
        segments_data = list(_session.get("segments_data", []))
        started_at = _session.get("started_at", "")

    if not segments_data:
        return None

    return _build_pdf_report(segments_data, started_at, is_master=False)


def generate_master_report() -> Optional[str]:
    """Generate the 8-hour master report."""
    with _lock:
        if _session is None:
            return None
        segments_data = list(_session.get("segments_data", []))
        started_at = _session.get("started_at", "")

    if not segments_data:
        return None

    return _build_pdf_report(segments_data, started_at, is_master=True)


def _gemini_live_overview(global_stats, phases, segments_data):
    """Generate Gemini AI executive summary for live crowd monitoring session."""
    if not _cr.GEMINI_API_KEY:
        return _cr._generate_event_narrative(global_stats, phases)

    import urllib.request, json as _json

    # Gather per-camera aggregate info
    cam_summaries = {}
    for seg in segments_data:
        for cam_id, cam_data in seg.get("per_camera_data", {}).items():
            name = cam_data.get("camera_name", cam_id[-8:])
            if name not in cam_summaries:
                cam_summaries[name] = {"counts": [], "peak": 0}
            cam_summaries[name]["counts"].extend(cam_data.get("counts", []))
            cam_summaries[name]["peak"] = max(
                cam_summaries[name]["peak"],
                cam_data.get("peak_count", 0),
            )

    cam_desc = "; ".join(
        f"{n}: avg {np.mean(d['counts']):.0f}, peak {d['peak']}"
        for n, d in cam_summaries.items() if d["counts"]
    ) or "No per-camera data"

    phase_desc = "; ".join(
        f"{p['condition']} ({_cr.format_ts(p['start_sec'])}-{_cr.format_ts(p['end_sec'])}, avg {p['avg']:.0f})"
        for p in phases[:8]
    ) or "No distinct phases"

    high_phases = [p for p in phases if p['condition'] in ('High', 'Critical')]
    risk_note = (
        f"{len(high_phases)} high-risk phase(s) with elevated congestion."
        if high_phases else "No critical congestion phases detected."
    )

    prompt = (
        f"Write a professional 200-300 word crowd safety executive summary for a LIVE multi-camera "
        f"monitoring session report.\n"
        f"- Session duration: {int(global_stats['duration_processed'] // 60)} minutes\n"
        f"- Cameras monitored: {len(cam_summaries)}\n"
        f"- Per-camera summary: {cam_desc}\n"
        f"- Peak persons (single capture): {global_stats['peak_count']}\n"
        f"- Average persons per capture: {global_stats['avg_count']:.1f}\n"
        f"- Crowd phases: {phase_desc}\n"
        f"- Risk: {risk_note}\n\n"
        f"Include: 1) Overall situation 2) Density/congestion assessment "
        f"3) Camera-specific observations 4) Safety status 5) Recommendations.\n"
        f"Write as a flowing paragraph. No bullets. No headers. 200-300 words."
    )

    try:
        body = _json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 512, "temperature": 0.45}
        }).encode()
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-2.5-flash:generateContent?key={_cr.GEMINI_API_KEY}")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = _json.loads(r.read())
        text = resp['candidates'][0]['content']['parts'][0]['text'].strip()
        print("[RT-CROWD] Gemini live overview generated.")
        return text
    except Exception as e:
        print(f"[RT-CROWD] Gemini overview error, using fallback: {e}")
        return _cr._generate_event_narrative(global_stats, phases)


def _generate_camera_heatmap(heatmap_acc, base_frame, boxes):
    """Generate CSRNet density heatmap overlay for a camera. Returns overlay image."""
    if base_frame is None:
        return None
    # Use CSRNet density model for report heatmaps
    heatmap_bgr, _, _ = generate_density_heatmap(base_frame)
    if heatmap_bgr is not None:
        overlay = overlay_heatmap(base_frame, heatmap_bgr, alpha=0.5)
    else:
        # Fallback to old accumulation-based method
        if heatmap_acc is None:
            return None
        if heatmap_acc.max() > 0:
            heatmap_norm = heatmap_acc / heatmap_acc.max()
        else:
            heatmap_norm = heatmap_acc
        heatmap_colored = cv2.applyColorMap(
            (heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(base_frame, 0.6, heatmap_colored, 0.4, 0)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay


def _make_trend_chart(counts, timestamps, title="Persons per Capture"):
    """Generate a trend chart from capture-interval counts. Returns temp file path."""
    fig, ax = plt.subplots(figsize=(5.5, 1.8))
    x_vals = list(range(len(counts)))
    ax.plot(x_vals, counts, color='#1e3a8a', linewidth=1.4)
    ax.fill_between(x_vals, counts, color='#1e3a8a', alpha=0.15)
    ax.set_title(title, fontsize=7, pad=4)
    ax.set_xlabel("Capture index", fontsize=6)
    ax.set_ylabel("Count", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)
    fig.tight_layout(pad=0.3)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(tmp.name, dpi=120)
    plt.close(fig)
    return tmp.name


def _draw_live_cover_page(pdf, global_stats, phases, total_duration, ai_summary,
                          started_at, num_segments, is_master, num_cameras):
    """Draw cover page matching the IRIS Crowd Analysis Report template."""
    pdf.add_page()

    # ── Title banner ──
    pdf.set_fill_color(10, 25, 75)
    pdf.rect(0, 0, 210, 50, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 22)
    title = "IRIS CROWD MASTER REPORT" if is_master else "IRIS CROWD ANALYSIS REPORT"
    pdf.cell(0, 20, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "", 11)
    sub = (f"Live Camera Monitoring | Generated: "
           f"{_now_ist().strftime('%d %b %Y, %H:%M:%S')}")
    pdf.cell(0, 8, sub, new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(15)
    pdf.set_text_color(0, 0, 0)

    # ── Event Overview box (draw background rect, then content, then re-draw over rect) ──
    start_y = pdf.get_y()

    # First pass: measure content height
    pdf.set_xy(15, start_y + 5)
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(0, 8, "Event Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(15)
    pdf.multi_cell(180, 5, ai_summary)

    y_stats_box = pdf.get_y() + 8

    peak_cong = _cr.congestion_score(global_stats['peak_count'], num_cameras=max(1, num_cameras))
    stats_items = [
        ("Duration", f"{int(total_duration // 60)}m {int(total_duration % 60)}s"),
        ("Peak Persons", str(global_stats['peak_count'])),
        ("Avg Persons", f"{global_stats['avg_count']:.1f}"),
        ("Peak Congestion", f"{peak_cong}/10"),
    ]
    box_w = 45
    # Draw stats boxes (first pass to measure)
    final_section_y = y_stats_box + 22

    # Draw the background rectangle
    pdf.set_draw_color(10, 25, 75)
    pdf.set_fill_color(240, 245, 255)
    pdf.set_y(start_y)
    pdf.rect(10, start_y, 190, final_section_y - start_y, 'FD')

    # Re-draw content over the rect
    pdf.set_xy(15, start_y + 5)
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(0, 8, "Event Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(15)
    pdf.multi_cell(180, 5, ai_summary)

    for i, (label, val) in enumerate(stats_items):
        x = 10 + i * box_w
        pdf.set_xy(x, y_stats_box)
        pdf.set_text_color(10, 25, 75)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(box_w - 2, 12, val, align="C")
        pdf.set_xy(x, y_stats_box + 12)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", "", 8)
        pdf.cell(box_w - 2, 7, label, align="C")

    pdf.set_y(final_section_y + 12)

    # ── Key Phases ──
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(0, 8, "Key Phases Identified:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(10, 25, 75)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    phase_labels = {
        "Clear": "Inactive Period", "Low": "Low Activity",
        "Medium": "Moderate Build-Up", "High": "High Congestion",
        "Critical": "Critical Saturation",
    }
    for ph in phases[:12]:
        row_y = pdf.get_y()
        if row_y > 265:
            pdf.add_page()
            row_y = pdf.get_y()
        rp, gp, bp = _cr.condition_color(ph['condition'])
        label = phase_labels.get(ph['condition'], ph['condition'])
        ts_s = _cr.format_ts(ph['start_sec'])
        ts_e = _cr.format_ts(ph['end_sec'])
        pdf.set_fill_color(rp, gp, bp)
        pdf.rect(10, row_y + 1, 4, 5, 'F')
        pdf.set_xy(17, row_y)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(90, 7, f"{label}  ({ts_s} - {ts_e})", 0, 0)
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 7, f"Avg {ph['avg']:.0f} persons", 0, 1)
        pdf.ln(1)

    # ── Major Risks ──
    high_risk = [p for p in phases if p['condition'] in ("High", "Critical")]
    if high_risk:
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, "Major Risks Identified:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 0, 0)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        for i, ph in enumerate(high_risk[:5]):
            ts_s = _cr.format_ts(ph['start_sec'])
            ts_e = _cr.format_ts(ph['end_sec'])
            cong_val = _cr.congestion_score(ph['avg'])
            free_val = _cr.free_space_pct(ph['avg'])
            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(200, 0, 0)
            pdf.cell(5, 6, f"{i + 1}.", 0, 0)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("helvetica", "", 10)
            pdf.cell(0, 6,
                     f"{ph['condition']} Density Risk ({ts_s} - {ts_e}): "
                     f"Congestion {cong_val:.1f}/10, Free space ~{free_val}%.",
                     0, 1)
            pdf.ln(1)


def _draw_camera_page(pdf, cam_name, cam_data, seg_idx, total_segs,
                      seg_start, seg_end, analysis):
    """Draw a per-camera segment page matching the IRIS template layout."""
    pdf.add_page()

    cam_counts = cam_data.get("counts", [])
    avg_count = cam_data.get("avg_count", 0)
    peak_count = cam_data.get("peak_count", 0)
    num_cameras = cam_data.get("num_cameras", 1)

    cong = _cr.congestion_score(avg_count, peak=peak_count, num_cameras=num_cameras)
    cond = _cr.condition_from_score(cong)
    fl = _cr.flow_direction(cam_counts) if len(cam_counts) >= 4 else "Stable"
    predicted = _cr.predict_next_window(cam_counts)
    pred_cond = _cr.condition_from_score(_cr.congestion_score(predicted))
    insights = _cr.insights_from_segment(cam_counts, cond, fl, predicted)
    # Add Gemini safety check insights from analysis
    weapon = analysis.get('weapon_detected', 'NO')
    fight = analysis.get('fight_collision_injury', 'NO')
    wrongful = analysis.get('wrongful_activity', 'NO')
    insights.append(f"Is anyone carrying any weapon: {weapon}")
    insights.append(f"Is anyone involved in a fight/collision/injury: {fight}")
    insights.append(f"Any wrongful activity/crime occurring: {wrongful}")
    alert_text, alert_level = _cr.safety_alerts(cond, pred_cond)

    r, g, b = _cr.condition_color(cond)
    HEADER_H = 18

    # ── Color header bar ──
    pdf.set_fill_color(r, g, b)
    pdf.rect(0, 0, 210, HEADER_H, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 9)
    header = (f"{cam_name}    |    Time: {seg_start} - {seg_end}    |    "
              f"Segment {seg_idx}/{total_segs}    |    "
              f"Condition: {cond.upper()}    |    "
              f"Congestion: {cong}/10    |    Flow: {fl}")
    pdf.set_xy(0, (HEADER_H - 9) / 2)
    pdf.cell(210, 9, header, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_y(HEADER_H)

    # ── Heatmap image (left side) ──
    heatmap_path = cam_data.get("heatmap_path")
    heatmap_y = HEADER_H + 3
    if heatmap_path and os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=10, y=heatmap_y, w=130, h=73)
        # Show person count for this heatmap frame
        heatmap_frame_count = cam_data.get("heatmap_frame_count", 0)
        pdf.set_xy(10, heatmap_y + 74)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(10, 25, 75)
        pdf.cell(130, 5, f"Persons detected in frame: {heatmap_frame_count}", align="C")

    # ── Right sidebar ──
    sidebar_x = 145
    sidebar_y = HEADER_H + 2

    # Condition badge
    pdf.set_xy(sidebar_x, sidebar_y)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(60, 10, cond.upper(), new_x="LMARGIN", new_y="NEXT", align="C", fill=True)

    # Congestion Score label
    pdf.set_xy(sidebar_x, sidebar_y + 12)
    pdf.set_fill_color(235, 235, 245)
    pdf.set_text_color(10, 25, 75)
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(60, 7, "Congestion Score", new_x="LMARGIN", new_y="NEXT", fill=True, align="C")

    # Score value
    pdf.set_xy(sidebar_x, sidebar_y + 19)
    pdf.set_fill_color(10, 25, 75)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(60, 12, f"{cong}/10", new_x="LMARGIN", new_y="NEXT", fill=True, align="C")

    # Crowd Insights
    pdf.set_xy(sidebar_x, sidebar_y + 33)
    pdf.set_font("helvetica", "B", 8)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(60, 5, "Crowd Insights", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(sidebar_x, pdf.get_y())

    for ins in insights:
        iy = pdf.get_y()
        pdf.set_xy(sidebar_x, iy)
        pdf.set_fill_color(219, 234, 254)
        pdf.set_text_color(30, 58, 138)
        pdf.set_font("helvetica", "", 7)
        pdf.multi_cell(60, 4, f"- {ins}", fill=True)
        pdf.ln(1)

    # NEXT prediction
    pdf.set_xy(sidebar_x, pdf.get_y() + 2)
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(34, 197, 94)
    pdf.set_text_color(0, 100, 0)
    pdf.set_font("helvetica", "B", 8)
    pdf.set_x(sidebar_x)
    pdf.cell(60, 5, "NEXT SEGMENT PREDICTION", fill=True)
    pdf.ln(5)
    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(sidebar_x)
    pdf.cell(60, 5, f"Predicted: ~{predicted} persons", fill=True)
    pdf.ln(5)
    pdf.set_x(sidebar_x)
    pdf.cell(60, 5, f"Forecast condition: {pred_cond}", fill=True)
    pdf.ln(5)

    # Safety alert
    sar, sag, sab = _cr.condition_color(alert_level)
    pdf.set_xy(sidebar_x, pdf.get_y() + 2)
    pdf.set_fill_color(sar, sag, sab)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 7)
    pdf.multi_cell(60, 4, f"SAFETY: {alert_text}", fill=True)

    # ── CROWD DYNAMICS & ENVIRONMENT ──
    gap_y = heatmap_y + 82
    pdf.set_xy(10, gap_y)
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(60, 5, "CROWD DYNAMICS & ENVIRONMENT", align="L")
    pdf.ln(6)

    visibility = analysis.get("visibility_score", 90)
    sentiment_raw = analysis.get("sentiment", "NEUTRAL")
    behavior = "Normal crowd flow" if avg_count < 50 else "Dense crowd activity"

    pdf.set_x(10)
    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(35, 5, f"Visibility Score: {visibility}%")
    pdf.cell(45, 5, f"Behavior: {behavior}")

    # Sentiment meter
    meter_x, meter_y = 92, gap_y + 6
    pdf.set_xy(meter_x, meter_y)
    pdf.set_font("helvetica", "B", 7)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(20, 4, "SENTIMENT:", align="L")

    bar_x, bar_w = meter_x + 18, 30
    pdf.set_fill_color(230, 230, 230)
    pdf.rect(bar_x, meter_y + 0.5, bar_w, 3, 'F')

    sent_map = {"NEUTRAL": 0.1, "CAUTIOUS": 0.35, "AGITATED": 0.65, "MOB": 0.9}
    sent_val = sent_map.get(sentiment_raw, 0.1)
    pdf.set_fill_color(34, 197, 94)
    pdf.rect(bar_x, meter_y + 0.5, max(1.5, sent_val * bar_w), 3, 'F')

    pdf.set_font("helvetica", "", 6)
    pdf.set_xy(bar_x, meter_y + 3.5)
    pdf.set_text_color(22, 101, 52)
    pdf.cell(bar_w / 2, 4, "Neutral", align="L")
    pdf.set_text_color(153, 27, 27)
    pdf.cell(bar_w / 2, 4, "Mob", align="R")

    # ── Processed Frames ──
    thumb_y = 115
    pdf.set_y(thumb_y - 3)
    pdf.set_font("helvetica", "B", 8)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(0, 5, "Processed Frames:", new_x="LMARGIN", new_y="NEXT")

    thumbnails = cam_data.get("thumbnails", [])
    thumb_w = 37
    for i, thumb_info in enumerate(thumbnails[:4]):
        tpath = thumb_info.get("path", "")
        tcount = thumb_info.get("count", 0)
        tts = thumb_info.get("timestamp", "")
        if tpath and os.path.exists(tpath):
            tx = 10 + i * (thumb_w + 2)
            pdf.image(tpath, x=tx, y=thumb_y + 2, w=thumb_w, h=21)
            pdf.set_xy(tx, thumb_y + 24)
            pdf.set_font("helvetica", "", 6)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(thumb_w, 4, tts, align="C", new_x="RIGHT", new_y="TOP")
            pdf.set_xy(tx, thumb_y + 28)
            pdf.set_fill_color(50, 50, 50)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(thumb_w, 4, f"{tcount} ppl", align="C", fill=True,
                     new_x="RIGHT", new_y="TOP")

    # ── Trend chart ──
    if cam_counts and len(cam_counts) > 1:
        chart_path = _make_trend_chart(
            cam_counts,
            cam_data.get("timestamps", []),
            f"Persons per Capture ({cam_name})",
        )
        chart_y = thumb_y + 36
        pdf.image(chart_path, x=10, y=chart_y, w=130, h=40)
        try:
            os.unlink(chart_path)
        except Exception:
            pass


def _build_pdf_report(segments_data: list, started_at: str, is_master: bool = False) -> str:
    """Build IRIS-style PDF crowd report matching the template format.

    Structure:
      - Cover page: Event Overview (Gemini), Key Metrics, Key Phases, Major Risks
      - Per camera per segment: Heatmap, Sidebar (congestion, insights, prediction,
        safety), Crowd Dynamics, Processed Frames, Trend Chart
    """
    report_id = uuid.uuid4().hex[:8]
    report_type = "master" if is_master else "ondemand"
    timestamp = _now_ist().strftime("%Y%m%d_%H%M%S")
    pdf_path = str(REPORTS_DIR / f"crowd_{report_type}_{timestamp}_{report_id}.pdf")

    # ── Compute global stats ──
    all_counts = []
    active_cameras = set()
    all_start_times = []
    all_end_times = []
    for seg in segments_data:
        all_counts.extend(seg.get("per_capture_counts", []))
        for cid in seg.get("per_camera_data", {}):
            active_cameras.add(cid)
        s, e = seg.get("segment_start", ""), seg.get("segment_end", "")
        if s:
            all_start_times.append(s)
        if e:
            all_end_times.append(e)

    # Duration = wall-clock span from earliest start to latest end.
    # Summing per-segment duration_sec is wrong when segments overlap.
    total_duration = 0
    if all_start_times and all_end_times:
        try:
            fmt = "%H:%M:%S"
            earliest = min(datetime.strptime(t, fmt) for t in all_start_times)
            latest = max(datetime.strptime(t, fmt) for t in all_end_times)
            total_duration = (latest - earliest).total_seconds()
        except ValueError:
            pass

    global_stats = {
        'peak_count': max(all_counts) if all_counts else 0,
        'avg_count': float(np.mean(all_counts)) if all_counts else 0,
        'duration_processed': total_duration,
    }

    per_second_avgs = []
    for seg in segments_data:
        per_second_avgs.extend(seg.get("per_capture_counts", []))
    phases = _cr.detect_phases(per_second_avgs, min_phase_sec=2)

    # ── Gemini event overview ──
    ai_summary = _gemini_live_overview(global_stats, phases, segments_data)

    # ── Build PDF ──
    pdf = _cr.IrisPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Cover Page ──
    _draw_live_cover_page(
        pdf, global_stats, phases, total_duration, ai_summary,
        started_at, len(segments_data), is_master, len(active_cameras),
    )

    # ── Per-Camera Segment Pages ──
    total_segs = len(segments_data)
    for seg in segments_data:
        per_camera = seg.get("per_camera_data", {})
        analysis = seg.get("analysis", {})
        seg_idx = seg.get("segment_index", 0)
        seg_start = seg.get("segment_start", "")
        seg_end = seg.get("segment_end", "")

        if not per_camera:
            continue

        for cam_id, cam_data in per_camera.items():
            cam_name = cam_data.get("camera_name", "Unknown")
            if not cam_data.get("counts"):
                continue
            _draw_camera_page(
                pdf, cam_name, cam_data,
                seg_idx, total_segs,
                seg_start, seg_end, analysis,
            )

    pdf.output(pdf_path)
    print(f"[RT-CROWD] Report generated: {pdf_path}")
    return pdf_path


def get_csv_path() -> Optional[str]:
    path = _csv_path()
    return str(path) if path.exists() else None


# ── Hourly Metadata Persistence ──

def _hourly_metadata_path(date_str: str) -> Path:
    """Return path to hourly metadata JSON for a given date (YYYY-MM-DD)."""
    return HOURLY_DIR / f"hourly_{date_str}.json"


def _load_hourly_metadata(date_str: str) -> dict:
    """Load hourly metadata for a date. Returns dict keyed by hour slot (e.g. '10-11')."""
    p = _hourly_metadata_path(date_str)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_hourly_metadata(date_str: str, data: dict):
    """Atomically save hourly metadata for a date."""
    p = _hourly_metadata_path(date_str)
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(p)


def _append_segment_to_hourly(seg_record: dict):
    """Append a segment record to the hourly slot based on its start time.

    Each segment is assigned to exactly one slot — the hour in which it starts.
    This keeps hourly reports strictly within their 60-minute window.
    """
    seg_start = seg_record.get("segment_start", "")
    seg_end = seg_record.get("segment_end", "")
    if not seg_start or not seg_end:
        return

    try:
        start_dt = datetime.strptime(seg_start, "%H:%M:%S")
    except ValueError:
        try:
            start_dt = datetime.strptime(seg_start, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return

    date_str = _now_ist().strftime("%Y-%m-%d")
    hourly = _load_hourly_metadata(date_str)

    # Assign to the slot where the segment starts
    start_hour = start_dt.hour
    slot_key = f"{start_hour:02d}-{(start_hour + 1) % 24:02d}"

    # Strip non-serializable fields (heatmap_path references local files, keep them)
    # but strip thumbnails raw data to keep JSON small
    clean_record = dict(seg_record)
    per_cam = clean_record.get("per_camera_data", {})
    for cid in per_cam:
        cam = per_cam[cid]
        # Keep only serializable metadata, not raw frame data
        cam.pop("thumbnails", None)

    if slot_key not in hourly:
        hourly[slot_key] = {"segments": [], "first_seen": seg_start, "last_seen": seg_end}
    hourly[slot_key]["segments"].append(clean_record)
    hourly[slot_key]["last_seen"] = seg_end

    _save_hourly_metadata(date_str, hourly)
    print(f"[RT-CROWD] Hourly metadata updated: {date_str} slot {slot_key}")


def get_hourly_slots(date_str: Optional[str] = None) -> list:
    """Return list of available hourly slots for a date. Each slot has summary stats."""
    if not date_str:
        date_str = _now_ist().strftime("%Y-%m-%d")
    hourly = _load_hourly_metadata(date_str)
    slots = []
    for slot_key in sorted(hourly.keys()):
        slot_data = hourly[slot_key]
        segs = slot_data.get("segments", [])
        all_counts = []
        cameras = set()
        for seg in segs:
            all_counts.extend(seg.get("per_capture_counts", []))
            for cid in seg.get("per_camera_data", {}):
                cameras.add(cid)
        slots.append({
            "slot": slot_key,
            "segments": len(segs),
            "cameras": len(cameras),
            "avg_count": round(float(np.mean(all_counts)), 1) if all_counts else 0,
            "peak_count": max(all_counts) if all_counts else 0,
            "first_seen": slot_data.get("first_seen", ""),
            "last_seen": slot_data.get("last_seen", ""),
        })
    return slots


def generate_hourly_report(slot_key: str, date_str: Optional[str] = None) -> Optional[str]:
    """Generate a PDF report for a specific hourly slot (e.g. '16-17').

    Only includes segments whose start time falls within the slot hour.
    The report duration is capped at 60 minutes.
    """
    if not date_str:
        date_str = _now_ist().strftime("%Y-%m-%d")
    hourly = _load_hourly_metadata(date_str)
    slot_data = hourly.get(slot_key)
    if not slot_data or not slot_data.get("segments"):
        return None

    # Parse slot hour boundaries
    try:
        slot_start_hour = int(slot_key.split("-")[0])
    except (ValueError, IndexError):
        return None

    # Filter: only keep segments whose start time is within the slot hour
    filtered = []
    for seg in slot_data["segments"]:
        seg_start = seg.get("segment_start", "")
        try:
            dt = datetime.strptime(seg_start, "%H:%M:%S")
        except ValueError:
            try:
                dt = datetime.strptime(seg_start, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        if dt.hour == slot_start_hour:
            filtered.append(seg)

    if not filtered:
        return None

    started_at = f"{date_str} {slot_data.get('first_seen', '')}"
    return _build_pdf_report(filtered, started_at, is_master=False)


def get_available_dates() -> list:
    """Return list of dates that have hourly metadata available."""
    dates = []
    for p in sorted(HOURLY_DIR.glob("hourly_*.json")):
        date_str = p.stem.replace("hourly_", "")
        dates.append(date_str)
    return dates


# ── Monitor Loop ──

def _monitor_loop():
    """Main monitoring loop: capture → detect → accumulate → analyze per segment."""
    global _session

    # Load YOLO model
    try:
        model = _load_yolo()
    except Exception as e:
        with _lock:
            if _session:
                _session["status"] = "error"
                _session["message"] = f"YOLO model load failed: {e}"
        print(f"[RT-CROWD] YOLO load failed: {e}")
        return

    # Init Gemini
    try:
        _ensure_genai()
    except Exception as e:
        with _lock:
            if _session:
                _session["status"] = "error"
                _session["message"] = f"Gemini init failed: {e}"
        print(f"[RT-CROWD] Gemini init failed: {e}")
        return

    gemini_model = _forensics_mod._genai_module.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={"max_output_tokens": FORENSICS_MAX_OUTPUT_TOKENS, "temperature": 0.2},
    )

    with _lock:
        if _session:
            _session["message"] = f"Monitoring {len(CAMERAS)} cameras (10-min segments)..."

    with _lock:
        camera_ids = list(_session.get("selected_cameras", CAMERAS.keys()))
    start_time = time.time()
    segment_idx = 1
    segment_start_time = time.time()
    segment_counts = []  # per-capture total counts
    segment_frames_data = []  # frames data for Gemini analysis
    cam_index = 0

    # Per-camera tracking within each segment
    def _init_cam_segment_data():
        return {cid: {
            'counts': [],
            'timestamps': [],
            'heatmap_acc': None,
            'frame_dims': None,
            'base_frame': None,
            'last_boxes': [],
            'thumbnails': [],
        } for cid in camera_ids}

    cam_segment_data = _init_cam_segment_data()
    capture_idx_in_seg = 0
    # Thumbnail targets: save 4 thumbnails evenly spaced through the segment
    expected_captures = max(1, SEGMENT_DURATION_SEC // CAPTURE_INTERVAL)
    thumb_interval = max(1, expected_captures // 4)

    _init_csv()

    while not _stop_event.is_set():
        elapsed = time.time() - start_time

        # 8-hour auto-stop
        if elapsed >= MAX_RUNTIME_SEC:
            print("[RT-CROWD] 8-hour max runtime reached. Generating master report...")
            with _lock:
                if _session:
                    _session["message"] = "Max runtime reached. Generating master report..."
            master_path = generate_master_report()
            with _lock:
                if _session:
                    _session["status"] = "completed"
                    _session["message"] = f"8-hour session complete. Master report: {master_path}"
                    _session["master_report_path"] = master_path
            return

        # Re-read camera list each cycle to pick up dynamically added/removed cameras
        with _lock:
            camera_ids = list(_session.get("selected_cameras", CAMERAS.keys()))
        # Ensure new cameras have segment tracking data
        for cid in camera_ids:
            if cid not in cam_segment_data:
                cam_segment_data[cid] = {
                    'counts': [], 'timestamps': [], 'heatmap_acc': None,
                    'frame_dims': None, 'base_frame': None, 'last_boxes': [], 'thumbnails': [],
                }

        # Capture frames from all cameras in parallel
        capture_total_count = 0
        capture_frames = []

        capture_time_str = _now_ist().strftime("%H:%M:%S")

        # ── Parallel capture: grab frames from all cameras simultaneously ──
        captured = {}  # cid → frame (or None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(camera_ids), 16)) as pool:
            futures = {pool.submit(_capture_frame, cid): cid for cid in camera_ids}
            for fut in concurrent.futures.as_completed(futures):
                if _stop_event.is_set():
                    return
                cid = futures[fut]
                try:
                    captured[cid] = fut.result()
                except Exception:
                    captured[cid] = None

        # ── Sequential inference + tracking (GPU is single-threaded) ──
        for cid in camera_ids:
            if _stop_event.is_set():
                return

            frame = captured.get(cid)
            if frame is None:
                continue

            boxes, count = _detect_heads(model, frame)
            capture_total_count += count

            # Encode for Gemini (only keep last batch worth)
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FORENSICS_JPEG_QUALITY])
            capture_frames.append({
                "camera_id": cid,
                "camera_name": CAMERAS[cid],
                "capture_time": capture_time_str,
                "head_count": count,
                "frame_jpeg": buffer.tobytes(),
                "boxes": boxes,
            })

            # ── Per-camera tracking ──
            csd = cam_segment_data[cid]
            csd['counts'].append(count)
            csd['timestamps'].append(capture_time_str)

            # Accumulate heatmap
            h, w = frame.shape[:2]
            if csd['heatmap_acc'] is None:
                csd['heatmap_acc'] = np.zeros((h, w), dtype=np.float32)
                csd['frame_dims'] = (h, w)
            for (x1, y1, x2, y2) in boxes:
                cx = max(0, min(w - 1, (x1 + x2) // 2))
                cy = max(0, min(h - 1, (y1 + y2) // 2))
                cv2.circle(csd['heatmap_acc'], (cx, cy), 40, 1.0, -1)

            csd['base_frame'] = frame
            csd['last_boxes'] = boxes

            # Publish annotated frame + analysis for live dashboard
            _publish_cam_analysis(cid, frame, boxes, count, CAMERAS[cid])

            # Save thumbnails at evenly spaced intervals (4 per camera)
            if (capture_idx_in_seg % thumb_interval == 0
                    and len(csd['thumbnails']) < 4):
                thumb_frame = frame.copy()
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(thumb_frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 1)
                tpath = str(
                    FRAMES_DIR
                    / f"seg{segment_idx}_{cid[-8:]}_t{len(csd['thumbnails'])}.jpg"
                )
                cv2.imwrite(tpath, cv2.resize(thumb_frame, (320, 180)))
                csd['thumbnails'].append({
                    "path": tpath,
                    "count": count,
                    "timestamp": capture_time_str,
                })
                del thumb_frame

        segment_counts.append(capture_total_count)
        capture_idx_in_seg += 1

        # Keep the latest batch of frames for Gemini analysis
        if capture_frames:
            segment_frames_data = capture_frames  # overwrite with latest batch

        # Build per-camera count map for frontend
        per_camera_counts = {}
        for cf in capture_frames:
            per_camera_counts[cf['camera_id']] = cf['head_count']

        with _lock:
            if _session:
                _session["total_captures"] = _session.get("total_captures", 0) + 1
                _session["current_segment"] = segment_idx
                _session["per_camera_counts"] = per_camera_counts
                _session["last_total_count"] = capture_total_count

        # Check if segment is complete (10 minutes)
        segment_elapsed = time.time() - segment_start_time
        if segment_elapsed >= SEGMENT_DURATION_SEC:
            seg_start_str = datetime.fromtimestamp(segment_start_time, tz=IST).strftime("%H:%M:%S")
            seg_end_str = _now_ist().strftime("%H:%M:%S")

            # Compute segment stats
            seg_stats = {
                "total_count": sum(segment_counts),
                "avg_count": float(np.mean(segment_counts)) if segment_counts else 0,
                "peak_count": max(segment_counts) if segment_counts else 0,
                "num_captures": len(segment_counts),
                "segment_duration_min": segment_elapsed / 60,
            }

            # Run Gemini crowd analysis
            print(f"[RT-CROWD] Segment {segment_idx} complete ({seg_start_str}-{seg_end_str}). Running Gemini analysis...")
            analysis = _gemini_crowd_analysis(segment_frames_data, seg_stats, gemini_model)

            # ── Build per-camera data with heatmaps ──
            per_camera_data = {}
            for cid in camera_ids:
                csd = cam_segment_data[cid]
                if not csd['counts']:
                    continue

                # Generate heatmap overlay image
                heatmap_path = None
                if csd['heatmap_acc'] is not None and csd['base_frame'] is not None:
                    hmap_img = _generate_camera_heatmap(
                        csd['heatmap_acc'], csd['base_frame'], csd['last_boxes']
                    )
                    if hmap_img is not None:
                        hmap_file = FRAMES_DIR / f"seg{segment_idx}_{cid[-8:]}_heatmap.jpg"
                        cv2.imwrite(str(hmap_file), hmap_img)
                        heatmap_path = str(hmap_file)
                        del hmap_img

                per_camera_data[cid] = {
                    "camera_name": CAMERAS[cid],
                    "counts": list(csd['counts']),
                    "timestamps": list(csd['timestamps']),
                    "heatmap_path": heatmap_path,
                    "heatmap_frame_count": len(csd['last_boxes']),
                    "thumbnails": list(csd['thumbnails']),
                    "peak_count": max(csd['counts']) if csd['counts'] else 0,
                    "avg_count": float(np.mean(csd['counts'])) if csd['counts'] else 0,
                    "seg_avg_count": seg_stats.get("avg_count", 0),
                    "seg_peak_count": seg_stats.get("peak_count", 0),
                    "num_cameras": len(per_camera_data) + 1,
                }

            # Build segment record
            seg_record = {
                "segment_index": segment_idx,
                "segment_start": seg_start_str,
                "segment_end": seg_end_str,
                "duration_sec": segment_elapsed,
                "stats": seg_stats,
                "analysis": analysis,
                "per_capture_counts": list(segment_counts),
                "per_camera_data": per_camera_data,
            }

            # Log to CSV
            _log_to_csv(
                segment_idx, seg_start_str, seg_end_str,
                seg_stats, analysis, len(camera_ids)
            )

            # Persist to hourly metadata for on-demand hourly reports
            _append_segment_to_hourly(seg_record)

            # Build event summary for frontend
            event_summary = {
                "segment": segment_idx,
                "time": f"{seg_start_str} - {seg_end_str}",
                "risk": analysis.get("overall_risk", "LOW"),
                "density": analysis.get("crowd_density", "N/A"),
                "sentiment": analysis.get("sentiment", "NEUTRAL"),
                "movement": analysis.get("crowd_movement", ""),
                "weapon": analysis.get("weapon_detected", "NO"),
                "fight": analysis.get("fight_collision_injury", "NO"),
                "wrongful": analysis.get("wrongful_activity", "NO"),
                "visibility": analysis.get("visibility_score", 90),
                "predicted_next": analysis.get("predicted_count_next_segment", 0),
                "safety": analysis.get("safety_precaution", ""),
                "avg_count": seg_stats["avg_count"],
                "peak_count": seg_stats["peak_count"],
                "per_camera": analysis.get("per_camera_summary", []),
            }

            with _lock:
                if _session:
                    _session["segments_completed"] = segment_idx
                    _session["recent_analysis"] = analysis
                    _session["events"].append(event_summary)
                    _session["segments_data"].append(seg_record)
                    _session["message"] = (
                        f"Segment {segment_idx} analyzed. "
                        f"Risk: {analysis.get('overall_risk', 'N/A')} | "
                        f"Density: {analysis.get('crowd_density', 'N/A')} | "
                        f"Avg: {seg_stats['avg_count']:.0f} persons"
                    )

            print(f"[RT-CROWD] Segment {segment_idx}: Risk={analysis.get('overall_risk')} "
                  f"Density={analysis.get('crowd_density')} Avg={seg_stats['avg_count']:.0f}")

            # Reset for next segment
            segment_idx += 1
            segment_start_time = time.time()
            segment_counts = []
            segment_frames_data = []
            cam_segment_data = _init_cam_segment_data()
            capture_idx_in_seg = 0

        # Wait before next capture round
        _stop_event.wait(CAPTURE_INTERVAL)

    # Clean up live frame buffers
    clear_live_cam_frames()

    # If stopped manually, still offer master report if enough data
    with _lock:
        if _session and _session.get("segments_data"):
            _session["message"] = "Session stopped. Report data available."
