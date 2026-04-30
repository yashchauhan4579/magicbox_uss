"""
Advanced Crowd Analytics Report Engine
Generates IRIS-style PDF report with:
- Event overview & key phases summary
- Per-60-second segment analysis with heatmaps, thumbnails, metrics, insights, predictions
- Rule-based summarization (fully local), with optional Gemini API for narratives
"""

import cv2
import os
import time
import uuid
import threading
import math
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            if _line.startswith("GEMINI_API_KEY"):
                GEMINI_API_KEY = _line.split("=", 1)[1].strip().strip('"')
                break

_BACKEND_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _BACKEND_DIR / "models"
MODEL_PATH_CROWD_YOLO = str(_MODELS_DIR / "best_head.pt")

CROWD_REPORT_DIR = _BACKEND_DIR / "data" / "crowd_reports"
CROWD_REPORT_DIR.mkdir(parents=True, exist_ok=True)

CROWD_REPORT_UPLOAD_DIR = _BACKEND_DIR / "data" / "uploads" / "crowd_reports"
CROWD_REPORT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Job tracking
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Live frame streaming: stores the latest JPEG-encoded annotated frame per job
_live_frames: Dict[str, bytes] = {}
_live_frames_lock = threading.Lock()
_live_frame_seq: Dict[str, int] = {}
_JPEG_QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 70]


MODEL_PATH_CROWD_CCN = str(_MODELS_DIR / "crowd-model.pth")

# Cache for Gemini road mask per video (avoid re-querying for every frame)
_road_mask_cache: Dict[str, np.ndarray] = {}


def _gemini_road_mask(frame, cache_key="default"):
    """Use Gemini Vision to identify road/open-ground regions in an aerial frame.
    Returns a binary mask (0-1 float) where 1 = road/open area, 0 = building/tree/structure."""
    import base64, json, urllib.request

    if cache_key in _road_mask_cache:
        return _road_mask_cache[cache_key]

    if not GEMINI_API_KEY:
        return None

    h, w = frame.shape[:2]
    # Resize for API efficiency
    small = cv2.resize(frame, (424, 240))
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    prompt = (
        "Analyze this aerial/drone image of a crowd gathering. Identify ALL areas where people/crowd "
        "are present or could gather — roads, streets, intersections, open grounds, plazas, sidewalks, "
        "and any open areas between buildings.\n\n"
        "IMPORTANT: Be GENEROUS with boundaries. Include the full width of roads including shoulders "
        "and adjacent open areas. The crowd fills the roads so include ALL areas where white-clothed "
        "people are visible from above.\n\n"
        "Return JSON: {\"road_regions\": [{\"points\": [[x1,y1],[x2,y2],...], \"label\": \"description\"}]}\n\n"
        "Coordinates are percentages (0-100) of image width (x) and height (y).\n"
        "Do NOT include building rooftops, dense tree canopy, or flyover structures.\n"
        "Return ONLY valid JSON, no markdown."
    )

    try:
        body = json.dumps({
            'contents': [{'parts': [
                {'text': prompt},
                {'inline_data': {'mime_type': 'image/jpeg', 'data': img_b64}}
            ]}],
            'generationConfig': {'maxOutputTokens': 1024, 'temperature': 0.2}
        }).encode()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read())

        text = resp['candidates'][0]['content']['parts'][0]['text'].strip()
        # Clean markdown wrapping if present
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
            text = text.rsplit('```', 1)[0]
        data = json.loads(text)

        # Build mask from polygon regions
        mask = np.zeros((h, w), dtype=np.float32)
        for region in data.get('road_regions', []):
            pts = region.get('points', [])
            if len(pts) >= 3:
                poly = np.array([[int(p[0] * w / 100), int(p[1] * h / 100)] for p in pts], dtype=np.int32)
                cv2.fillPoly(mask, [poly], 1.0)

        # Dilate generously to cover full road width + crowd spillover
        kernel = np.ones((25, 25), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (41, 41), 15.0)
        mask = np.clip(mask, 0, 1.0)

        # Ensure minimum coverage — if mask is too small, Gemini drew too tight
        coverage = mask.mean()
        if coverage < 0.15:
            print(f"[CrowdReport][Gemini] Road mask too small ({100*coverage:.1f}%), expanding...")
            mask = cv2.dilate(mask, np.ones((35, 35), np.uint8), iterations=3)
            mask = cv2.GaussianBlur(mask, (51, 51), 20.0)
            mask = np.clip(mask, 0, 1.0)

        _road_mask_cache[cache_key] = mask
        print(f"[CrowdReport][Gemini] Road mask generated: {int(mask.sum())} road pixels ({100*mask.mean():.1f}% of frame)")
        return mask

    except Exception as e:
        print(f"[CrowdReport][Gemini] Road mask failed: {e}")
        return None


def _gemini_segment_insights(frame, seg_data):
    """Use Gemini Vision to analyze a segment frame and provide accurate crowd insights."""
    import base64, json, urllib.request

    if not GEMINI_API_KEY:
        return None

    small = cv2.resize(frame, (424, 240))
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    count = seg_data.get('count', 0)
    congestion = seg_data.get('congestion', 0)
    condition = seg_data.get('condition', 'Unknown')
    yolo_heads = seg_data.get('yolo_head_count', 0)

    prompt = (
        f"Analyze this crowd surveillance image. AI density estimation: ~{count} persons. "
        f"Independent head detection found {yolo_heads} individual heads. "
        f"Congestion level {congestion}/10 ({condition} condition).\n\n"
        f"Provide a JSON response with ALL these fields:\n"
        f"1. \"narrative\": 20-30 word professional crowd safety observation\n"
        f"2. \"insights\": list of 4-5 specific observations (crowd density, movement patterns, "
        f"safety concerns, police/security presence, weapons, fights, road conditions)\n"
        f"3. \"predicted_trend\": \"increasing\", \"stable\", or \"decreasing\"\n"
        f"4. \"safety_level\": \"safe\", \"caution\", \"warning\", or \"critical\"\n"
        f"5. \"safety_note\": one-line safety recommendation\n"
        f"6. \"visibility_pct\": integer 0-100 camera/scene visibility quality\n"
        f"7. \"sentiment\": float 0.0 (calm/neutral) to 1.0 (agitated/mob-like)\n"
        f"8. \"behavior_description\": short crowd behavior description (e.g. 'Prayer gathering, orderly')\n"
        f"9. \"predicted_count_next_60s\": integer predicted person count for next 60 seconds\n"
        f"10. \"predicted_condition_next_60s\": \"Clear\", \"Low\", \"Medium\", \"High\", or \"Critical\"\n\n"
        f"Return ONLY valid JSON, no markdown."
    )

    try:
        body = json.dumps({
            'contents': [{'parts': [
                {'text': prompt},
                {'inline_data': {'mime_type': 'image/jpeg', 'data': img_b64}}
            ]}],
            'generationConfig': {'maxOutputTokens': 512, 'temperature': 0.3}
        }).encode()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read())

        text = resp['candidates'][0]['content']['parts'][0]['text'].strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
            text = text.rsplit('```', 1)[0]
        return json.loads(text)

    except Exception as e:
        print(f"[CrowdReport][Gemini] Segment insights failed: {e}")
        return None


def _load_crowd_model():
    """Load CCN density model + YOLO head model for hybrid crowd detection."""
    import torch
    from crowd import CrowdCounter
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    counter = CrowdCounter(MODEL_PATH_CROWD_CCN, device=device)
    yolo = YOLO(MODEL_PATH_CROWD_YOLO, task="detect")
    if device == "cuda":
        yolo.to(device)
    return (counter, yolo), device


# ---------------------------------------------------------------------------
# YOLO head detection + hybrid density
# ---------------------------------------------------------------------------

def _yolo_head_detect(yolo_model, frame):
    """Run YOLO head detection with auto-tiling for drone footage.

    Returns (detections, confidence_map):
      - detections: list of (cx, cy, w, h) tuples
      - confidence_map: float32 array (same size as frame) with Gaussian blobs at detections
    """
    h, w = frame.shape[:2]

    # Single-pass first
    results = yolo_model(frame, conf=0.12, iou=0.5, verbose=False, imgsz=480)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if r.names[cls] in ['person', 'people', 'head']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append(((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1))

    # If single-pass found nothing, try tiled for high-altitude drone footage
    if len(detections) == 0:
        tile_size, step = 240, 160
        all_boxes = []
        for ty in range(0, h, step):
            for tx in range(0, w, step):
                ty2 = min(ty + tile_size, h)
                tx2 = min(tx + tile_size, w)
                tile = frame[ty:ty2, tx:tx2]
                if tile.shape[0] < 50 or tile.shape[1] < 50:
                    continue
                tres = yolo_model(tile, conf=0.12, iou=0.5, verbose=False, imgsz=640)
                for r in tres:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if r.names[cls] in ['person', 'people', 'head']:
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                            all_boxes.append([bx1 + tx, by1 + ty, bx2 + tx, by2 + ty, float(box.conf[0])])
        if all_boxes:
            arr = np.array(all_boxes)
            indices = cv2.dnn.NMSBoxes(arr[:, :4].tolist(), arr[:, 4].tolist(), 0.12, 0.4)
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = arr[i, :4].astype(int)
                    detections.append(((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1))

    # Build confidence map with Gaussian blobs at each detection
    conf_map = np.zeros((h, w), dtype=np.float32)
    for (cx, cy, bw, bh) in detections:
        sigma = max(15, int(max(bw, bh) * 1.5))
        # Place Gaussian blob
        size = sigma * 3
        y_lo = max(0, cy - size)
        y_hi = min(h, cy + size)
        x_lo = max(0, cx - size)
        x_hi = min(w, cx + size)
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        conf_map[y_lo:y_hi, x_lo:x_hi] = np.maximum(conf_map[y_lo:y_hi, x_lo:x_hi], g)

    return detections, conf_map


def _hybrid_density(ccn_density, yolo_conf_map, yolo_detections, road_mask=None):
    """Combine CCN density with YOLO head confidence to calibrate accuracy.

    - Boost density where both CCN and YOLO agree (up to 50%)
    - Attenuate density where YOLO finds nothing within road areas (by 60%)
    - Skip attenuation if YOLO found 0 detections (altitude issue)
    """
    if len(yolo_detections) == 0:
        return ccn_density  # YOLO can't see at this altitude, trust CCN alone

    h, w = ccn_density.shape[:2]
    if yolo_conf_map.shape[:2] != (h, w):
        yolo_conf_map = cv2.resize(yolo_conf_map, (w, h), interpolation=cv2.INTER_LINEAR)

    hybrid = ccn_density.copy()
    ccn_max = ccn_density.max()
    if ccn_max < 1e-8:
        return hybrid

    ccn_norm = ccn_density / ccn_max

    # Boost: where YOLO confirms crowd, amplify CCN density
    boost_mask = (yolo_conf_map > 0.1) & (ccn_norm > 0.02)
    boost_factor = 1.0 + 0.5 * yolo_conf_map
    hybrid[boost_mask] = ccn_density[boost_mask] * boost_factor[boost_mask]

    # Attenuate: within road areas where YOLO sees nothing, reduce CCN density
    if road_mask is not None:
        if road_mask.shape[:2] != (h, w):
            rm = cv2.resize(road_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            rm = road_mask
        no_yolo = (yolo_conf_map < 0.05) & (ccn_norm > 0.1) & (rm > 0.5)
        hybrid[no_yolo] *= 0.7  # reduce by 30% (gentle — YOLO misses at altitude)

    return hybrid


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def gemini_video_summary(global_stats, phases):
    if not GEMINI_API_KEY:
        return _generate_event_narrative(global_stats, phases)

    import urllib.request, json

    phase_desc = "; ".join([
        f"{p['condition']} activity ({format_ts(p['start_sec'])}-{format_ts(p['end_sec'])}, avg {p['avg']:.0f} persons)"
        for p in phases[:8]
    ]) if phases else "No distinct phases detected"

    high_phases = [p for p in phases if p['condition'] in ('High', 'Critical')]
    risk_note = (
        f"There were {len(high_phases)} high-risk phase(s) with elevated congestion."
        if high_phases else "No critical congestion phases were detected."
    )

    prompt = (
        f"Write a professional, detailed 200-300 word crowd safety executive summary for a CCTV footage analysis report. "
        f"The recording covers a window with the following data:\n"
        f"- Peak persons detected: {global_stats['peak_count']}\n"
        f"- Average persons per frame: {global_stats['avg_count']:.1f}\n"
        f"- Total recording duration: {int(global_stats['duration_processed']//60)} minutes {int(global_stats['duration_processed']%60)} seconds\n"
        f"- Crowd phases: {phase_desc}\n"
        f"- Risk note: {risk_note}\n\n"
        f"Write from the perspective of a professional crowd safety analyst. The summary must include:\n"
        f"1. Opening sentence describing the overall crowd situation.\n"
        f"2. Density and congestion assessment (low/medium/high and what it implies).\n"
        f"3. Observed behavioral patterns across the recording duration.\n"
        f"4. Safety status and any risks identified.\n"
        f"5. Operational recommendations for crowd management.\n"
        f"Write as a single flowing paragraph. No bullet points. No headers. 200-300 words only."
    )

    try:
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.45}
        }).encode()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read())

        text = resp['candidates'][0]['content']['parts'][0]['text'].strip()
        if len(text) < 100:
            print(f"[CrowdReport][Gemini] Response too short ({len(text)} chars), using local fallback")
            return _generate_event_narrative(global_stats, phases)
        print("[CrowdReport][Gemini] AI executive summary generated.")
        return text
    except Exception as e:
        print(f"[CrowdReport][Gemini] API error, using local fallback: {e}")
        return _generate_event_narrative(global_stats, phases)


def gemini_segment_summary(seg_data):
    if not GEMINI_API_KEY:
        return _rule_based_narrative(seg_data)

    import urllib.request, json

    prompt = (
        f"In exactly 20-30 words, write a professional crowd safety observation for a 60-second CCTV segment. "
        f"Stats: {seg_data['count']} persons, congestion {seg_data['congestion']}/10, "
        f"crowd flow is {seg_data['flow'].lower()}, condition: {seg_data['condition']}. "
        f"Be specific and professional. No bullet points."
    )

    try:
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 256, "temperature": 0.5}
        }).encode()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())

        return resp['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception:
        return _rule_based_narrative(seg_data)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def congestion_score(count, max_expected=80, peak=None, num_cameras=1):
    """Congestion score (0-10) based on per-camera person count.

    Thresholds (per camera):
      < 12 people  → Clear    (score 0 - 3.0)
      < 25 people  → Low      (score 3.0 - 5.5)
      < 40 people  → Medium   (score 5.6 - 7.0)
      < 65 people  → High     (score 7.1 - 8.2)
      >= 65 people → Critical (score 8.3 - 10)
    """
    ppc = count / max(1, num_cameras)  # per-camera person count

    if ppc < 12:
        # 0 → 0.0, 12 → 3.0  (linear within Clear band)
        score = (ppc / 12.0) * 3.0
    elif ppc < 25:
        # 12 → 3.0, 25 → 5.5  (linear within Low band)
        score = 3.0 + ((ppc - 12) / 13.0) * 2.5
    elif ppc < 40:
        # 25 → 5.6, 40 → 7.0  (linear within Medium band)
        score = 5.6 + ((ppc - 25) / 15.0) * 1.4
    elif ppc < 65:
        # 40 → 7.1, 65 → 8.2  (linear within High band)
        score = 7.1 + ((ppc - 40) / 25.0) * 1.1
    else:
        # 65 → 8.3, 100+ → 10  (linear within Critical band, caps at 10)
        score = 8.3 + min(1.7, ((ppc - 65) / 35.0) * 1.7)

    return round(min(10.0, max(0.0, score)), 1)


def free_space_pct(count, max_capacity=100, num_cameras=1):
    ppc = count / max(1, num_cameras)
    return max(0, round(100 - (ppc / max_capacity) * 100))


def condition_from_score(score):
    if score >= 8.3: return "Critical"
    elif score >= 7.1: return "High"
    elif score >= 5.6: return "Medium"
    elif score >= 3.0: return "Low"
    return "Clear"

def condition_color(condition):
    return {
        "Clear":    (34, 197, 94),
        "Low":      (132, 204, 22),
        "Medium":   (234, 179, 8),
        "High":     (234, 88, 12),
        "Critical": (220, 38, 38),
    }.get(condition, (100, 100, 100))

def flow_direction(counts):
    if len(counts) < 4: return "Stable"
    trend = np.polyfit(range(len(counts)), counts, 1)[0]
    if trend > 1.0: return "Building"
    elif trend < -1.0: return "Dispersing"
    return "Stable"

def detect_phases(per_second_avg, min_phase_sec=2):
    phases = []
    if not per_second_avg:
        return phases
    current_cond = condition_from_score(congestion_score(per_second_avg[0]))
    start = 0
    for i, val in enumerate(per_second_avg):
        cond = condition_from_score(congestion_score(val))
        if cond != current_cond:
            if (i - start) >= min_phase_sec:
                phases.append({
                    'start_sec': start,
                    'end_sec': i - 1,
                    'condition': current_cond,
                    'avg': float(np.mean(per_second_avg[start:i]))
                })
            start = i
            current_cond = cond
    if (len(per_second_avg) - start) >= min_phase_sec:
        phases.append({
            'start_sec': start,
            'end_sec': len(per_second_avg) - 1,
            'condition': current_cond,
            'avg': float(np.mean(per_second_avg[start:]))
        })
    return phases

def format_ts(seconds_offset):
    h, r = divmod(int(seconds_offset), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def predict_next_window(counts):
    if len(counts) < 4:
        return counts[-1] if counts else 0
    X = np.array(range(len(counts))).reshape(-1, 1)
    y = np.array(counts)
    model = LinearRegression().fit(X, y)
    next_val = max(0, int(model.predict([[len(counts) + len(counts) // 2]])[0]))
    return next_val

def insights_from_segment(seg_counts, condition, flow, predicted):
    insights = []
    if flow == "Building":
        insights.append("Crowd movement is showing signs of convergence.")
    elif flow == "Dispersing":
        insights.append("Crowd movement is stable and dispersing.")
    else:
        insights.append("Crowd movement is stable with no sudden flows.")

    if condition in ("High", "Critical"):
        insights.append("Crowd density is at elevated levels.")
    elif condition == "Medium":
        insights.append("Crowd density is at moderate levels.")
    else:
        insights.append("Crowd density is within safe parameters.")

    pred_cond = condition_from_score(congestion_score(predicted))
    insights.append(f"Forecast: Count may reach ~{predicted} ({pred_cond}).")
    return insights

def safety_alerts(condition, predicted_condition):
    if condition == "Critical":
        return "CRITICAL: Immediate crowd control required. Crush risk elevated.", "Critical"
    elif condition == "High":
        return "ALERT: High congestion. Dispatch personnel to manage flow.", "High"
    elif predicted_condition in ("High", "Critical"):
        return "CAUTION: Crowd building. Proactive deployment recommended.", "Medium"
    elif condition == "Medium":
        return "MONITOR: Density is moderate. Continue observation.", "Low"
    return "No active safety alerts.", "Clear"


def _rule_based_narrative(d):
    cond = d['condition']
    flow = d['flow']
    count = d['count']
    free = d['free_space']
    cong = d['congestion']

    flow_text = {
        "Building": "The crowd density is increasing, indicating a convergence of people into this zone.",
        "Dispersing": "Crowd is moving out of the area in a controlled manner.",
        "Stable": "Movement is steady with no significant changes in density."
    }.get(flow, "")

    cond_text = {
        "Critical": f"Situation is CRITICAL. With only {free}% free space and a congestion score of {cong}/10, the area is at maximum safe capacity.",
        "High": f"Density is HIGH. Congestion at {cong}/10 with {free}% free space remaining.",
        "Medium": f"Density is at a moderate level. Approximately {count} persons are present with {free}% free space.",
        "Low": f"Low density with {count} persons and ample free space ({free}%).",
        "Clear": f"Area is clear. Only {count} persons detected with {free}% free space."
    }.get(cond, f"{count} persons observed in this segment.")

    return f"{cond_text} {flow_text}"


def _generate_event_narrative(stats, phases):
    peak = stats['peak_count']
    avg = stats['avg_count']
    high_phases = [p for p in phases if p['condition'] in ('High', 'Critical')]
    mid_phases = [p for p in phases if p['condition'] == 'Medium']

    if high_phases:
        risk_summary = f"Critical congestion was observed during {len(high_phases)} phase(s), with density reaching {congestion_score(peak):.1f}/10 at peak."
    else:
        risk_summary = "No critical congestion was observed. Conditions remained manageable throughout."

    build_summary = f"An average of {avg:.0f} persons were present at any given time, peaking at {peak} individuals."

    if high_phases and mid_phases:
        phase_desc = "The event transitioned through moderate and high-density phases, with periods of calm dispersal in between."
    elif high_phases:
        phase_desc = "The crowd exhibited sustained high-density conditions throughout the recording, indicating significant gathering activity."
    elif mid_phases:
        phase_desc = "The crowd maintained a moderate, controlled presence throughout most of the recorded period."
    else:
        phase_desc = "Crowd levels remained generally low and manageable across the duration."

    return f"{build_summary} {phase_desc} {risk_summary}"


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def finalize_heatmap(heatmap_acc, base_frame, mid_boxes, width, height):
    if heatmap_acc.max() > 0:
        heatmap_norm = heatmap_acc / heatmap_acc.max()
    else:
        heatmap_norm = heatmap_acc
    heatmap_colored = cv2.applyColorMap((75 + heatmap_norm * 180).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_frame, 0.5, heatmap_colored, 0.5, 0)
    for (x1, y1, x2, y2) in mid_boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, "Person", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return overlay


def finalize_heatmap_density(heatmap_acc, base_frame, width, height):
    """Build heatmap overlay matching irisv3 crowd-worker style:
    Full JET colormap with simple 50/50 blend. Blue tint comes from JET's low-density mapping."""
    h, w = base_frame.shape[:2]
    if heatmap_acc.shape[:2] != (h, w):
        heatmap_acc = cv2.resize(heatmap_acc, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_acc = np.maximum(heatmap_acc, 0)

    min_val = float(heatmap_acc.min())
    max_val = float(heatmap_acc.max())
    if max_val <= min_val or max_val == 0:
        return base_frame.copy()

    # Normalize (same as irisv3: plt.Normalize(vmin, vmax))
    heatmap_norm = np.clip((heatmap_acc - min_val) / (max_val - min_val), 0.0, 1.0)

    # Full JET colormap — blue on low density naturally tints background
    heatmap_colored = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Simple 50/50 blend (same as irisv3: cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0))
    overlay = cv2.addWeighted(base_frame, 0.5, heatmap_colored, 0.5, 0)
    return overlay


def _suppress_non_crowd(frame, density_map, yolo_model=None, gemini_road_mask=None, _precomputed_yolo=None):
    """Keep density only on road/open areas where crowd gathers.

    Uses Gemini Vision road mask (if available) as primary gate, then falls back
    to color/edge-based heuristics for remaining noise.
    _precomputed_yolo: optional (detections, conf_map) tuple to avoid double YOLO inference.
    """
    d_h, d_w = density_map.shape[:2]
    cleaned = np.maximum(density_map, 0).astype(np.float32)

    # --- Primary: Gemini road mask (most accurate) ---
    if gemini_road_mask is not None:
        gm_h, gm_w = gemini_road_mask.shape[:2]
        if gm_h != d_h or gm_w != d_w:
            road_mask = cv2.resize(gemini_road_mask, (d_w, d_h), interpolation=cv2.INTER_LINEAR)
        else:
            road_mask = gemini_road_mask
        cleaned = cleaned * road_mask
    else:
        # Fallback: color-based road detection
        frame_resized = cv2.resize(frame, (d_w, d_h))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]

        not_road = np.zeros((d_h, d_w), dtype=np.float32)
        green = cv2.inRange(hsv, (30, 35, 20), (90, 255, 255)).astype(np.float32) / 255.0
        not_road = np.maximum(not_road, green)
        blue = cv2.inRange(hsv, (90, 40, 40), (135, 255, 255)).astype(np.float32) / 255.0
        not_road = np.maximum(not_road, blue)
        red_lo = cv2.inRange(hsv, (0, 50, 40), (15, 255, 255)).astype(np.float32) / 255.0
        red_hi = cv2.inRange(hsv, (165, 50, 40), (180, 255, 255)).astype(np.float32) / 255.0
        not_road = np.maximum(not_road, np.maximum(red_lo, red_hi))
        dark = (v_ch < 55).astype(np.float32)
        not_road = np.maximum(not_road, dark)
        high_sat = (s_ch.astype(np.float32) > 90).astype(np.float32) * 0.6
        not_road = np.maximum(not_road, high_sat)
        edges = cv2.Canny(gray, 50, 150)
        edge_kernel = np.ones((19, 19), np.float32) / (19 * 19)
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, edge_kernel)
        edge_mask = np.clip((edge_density - 18) / 35.0, 0, 1.0) * 0.7
        not_road = np.maximum(not_road, edge_mask)
        not_road = cv2.GaussianBlur(not_road, (15, 15), 4.0)
        cleaned = cleaned * np.clip(1.0 - not_road, 0, 1.0)

    # --- Hybrid: YOLO head validation (boost/attenuate CCN density) ---
    if _precomputed_yolo is not None:
        yolo_dets, yolo_conf = _precomputed_yolo
    elif yolo_model is not None:
        yolo_dets, yolo_conf = _yolo_head_detect(yolo_model, frame)
    else:
        yolo_dets, yolo_conf = [], None
    if yolo_conf is not None and len(yolo_dets) > 0:
        cleaned = _hybrid_density(cleaned, yolo_conf, yolo_dets,
                                   road_mask=gemini_road_mask)

    # Always: remove small isolated blobs
    peak = float(cleaned.max())
    if peak > 1e-8:
        active = (cleaned >= peak * 0.06).astype(np.uint8)
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
        min_area = max(30, int(0.002 * d_h * d_w))
        keep_mask = np.zeros_like(active)
        for i in range(1, nlabels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep_mask[labels == i] = 1
        cleaned = cleaned * keep_mask.astype(np.float32)

    count = float(np.maximum(cleaned, 0).sum())
    count *= 1.8
    return cleaned, int(round(count))


def _overlay_density(frame, density_map):
    """Overlay density on thumbnails — irisv3 style: JET colormap, 50/50 blend."""
    h, w = frame.shape[:2]
    d = cv2.resize(np.maximum(density_map, 0), (w, h), interpolation=cv2.INTER_CUBIC)
    min_val, max_val = float(d.min()), float(d.max())
    if max_val <= min_val or max_val == 0:
        return frame
    d_norm = np.clip((d - min_val) / (max_val - min_val), 0, 1)
    heatmap = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

class IrisPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font('helvetica', 'I', 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f'IRIS Crowd Intelligence System | Confidential | Page {self.page_no()}', 0, 0, 'C')


def draw_cover_page(pdf, stats, phases, total_duration, ai_summary=""):
    pdf.add_page()

    pdf.set_fill_color(10, 25, 75)
    pdf.rect(0, 0, 210, 50, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 22)
    pdf.cell(0, 20, "IRIS CROWD ANALYSIS REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 8, f"Automated Crowd Intelligence | Generated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(15)
    pdf.set_text_color(0, 0, 0)

    start_y = pdf.get_y()

    pdf.set_xy(15, start_y + 5)
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(0, 8, "Event Overview", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(15)
    pdf.multi_cell(180, 5, ai_summary)

    y_stats_box = pdf.get_y() + 8

    stats_items = [
        ("Duration", f"{int(total_duration // 60)}m {int(total_duration % 60)}s"),
        ("Peak Persons", str(stats['peak_count'])),
        ("Avg Persons", f"{stats['avg_count']:.1f}"),
        ("Peak Congestion", f"{congestion_score(stats['peak_count'])}/10"),
    ]
    box_w = 45
    for i, (label, val) in enumerate(stats_items):
        x = 10 + i * box_w
        pdf.set_xy(x, y_stats_box)
        pdf.set_fill_color(10, 25, 75)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(box_w - 2, 12, val, new_x="RIGHT", new_y="TOP", align="C", fill=True)
        pdf.set_xy(x, y_stats_box + 12)
        pdf.set_fill_color(220, 230, 255)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", "", 8)
        pdf.cell(box_w - 2, 7, label, new_x="RIGHT", new_y="TOP", align="C", fill=True)

    final_section_y = y_stats_box + 22

    pdf.set_draw_color(10, 25, 75)
    pdf.set_fill_color(240, 245, 255)
    pdf.set_y(start_y)
    pdf.rect(10, start_y, 190, final_section_y - start_y, 'FD')

    # Redraw content over rect
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

    # Key Phases
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(0, 8, "Key Phases Identified:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(10, 25, 75)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    phase_labels = {
        "Clear": "Inactive Period",
        "Low": "Low Activity",
        "Medium": "Moderate Build-Up",
        "High": "High Congestion",
        "Critical": "Critical Saturation"
    }

    for ph in phases[:12]:
        row_y = pdf.get_y()
        r_ph, g_ph, b_ph = condition_color(ph['condition'])
        label = phase_labels.get(ph['condition'], ph['condition'])
        ts_start = format_ts(ph['start_sec'])
        ts_end = format_ts(ph['end_sec'])
        pdf.set_fill_color(r_ph, g_ph, b_ph)
        pdf.rect(10, row_y + 1, 4, 5, 'F')
        pdf.set_xy(17, row_y)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(90, 7, f"{label}  ({ts_start} - {ts_end})", 0, 0)
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 7, f"Avg {ph['avg']:.0f} persons", 0, 1)
        pdf.ln(1)

    high_risk_phases = [ph for ph in phases if ph['condition'] in ("High", "Critical")]
    if high_risk_phases:
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, "Major Risks Identified:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 0, 0)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        for i, ph in enumerate(high_risk_phases[:3]):
            ts_start = format_ts(ph['start_sec'])
            ts_end = format_ts(ph['end_sec'])
            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(200, 0, 0)
            pdf.cell(5, 6, f"{i+1}.", 0, 0)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("helvetica", "", 10)
            pdf.cell(0, 6, f"{ph['condition']} Density Risk ({ts_start} - {ts_end}): Congestion {congestion_score(ph['avg']):.1f}/10, Free space ~{free_space_pct(ph['avg'])}%.", 0, 1)
            pdf.ln(1)


def draw_segment_page(pdf, seg, seg_heatmap_path, thumb_paths_counts, seg_idx, total_segs):
    pdf.add_page()

    cond = seg['condition']
    r, g, b = condition_color(cond)

    HEADER_H = 18
    pdf.set_fill_color(r, g, b)
    pdf.rect(0, 0, 210, HEADER_H, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 9)
    header_text = (f"Time: {seg['ts_start']} - {seg['ts_end']}    |    "
                   f"Segment {seg_idx}/{total_segs}    |    "
                   f"Condition: {cond.upper()}    |    "
                   f"Congestion: {seg['congestion_score']}/10    |    "
                   f"Flow: {seg['flow']}")
    pdf.set_xy(0, (HEADER_H - 9) / 2)
    pdf.cell(210, 9, header_text, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_y(HEADER_H)

    pdf.ln(3)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("helvetica", "I", 9)
    pdf.multi_cell(130, 4, seg['narrative'])

    sidebar_x = 145
    sidebar_y = HEADER_H + 2

    pdf.set_xy(sidebar_x, sidebar_y)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(60, 10, cond.upper(), new_x="LMARGIN", new_y="NEXT", align="C", fill=True)

    pdf.set_xy(sidebar_x, sidebar_y + 12)
    pdf.set_fill_color(235, 235, 245)
    pdf.set_text_color(10, 25, 75)
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(60, 7, "Congestion Score", new_x="LMARGIN", new_y="NEXT", fill=True, align="C")

    pdf.set_xy(sidebar_x, sidebar_y + 19)
    pdf.set_fill_color(10, 25, 75)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(60, 12, f"{seg['congestion_score']}/10", new_x="LMARGIN", new_y="NEXT", fill=True, align="C")

    pdf.set_xy(sidebar_x, sidebar_y + 33)
    pdf.set_font("helvetica", "B", 8)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(60, 5, "Crowd Insights", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(sidebar_x, pdf.get_y())

    for insight in seg['insights']:
        iy = pdf.get_y()
        pdf.set_xy(sidebar_x, iy)
        pdf.set_fill_color(219, 234, 254)
        pdf.set_text_color(30, 58, 138)
        pdf.set_font("helvetica", "", 7)
        pdf.multi_cell(60, 4, f"- {insight}", fill=True)
        pdf.ln(1)

    pdf.set_xy(sidebar_x, pdf.get_y() + 2)
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(34, 197, 94)
    pdf.set_text_color(0, 100, 0)
    pdf.set_font("helvetica", "B", 8)
    pdf.set_x(sidebar_x)
    pdf.cell(60, 5, "NEXT 60s PREDICTION", fill=True)
    pdf.ln(5)

    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    pred_cond = condition_from_score(congestion_score(seg['predicted_count']))
    pdf.set_x(sidebar_x)
    pdf.cell(60, 5, f"Predicted: ~{seg['predicted_count']} persons", fill=True)
    pdf.ln(5)
    pdf.set_x(sidebar_x)
    pdf.cell(60, 5, f"Forecast condition: {pred_cond}", fill=True)
    pdf.ln(5)

    alert_text, alert_level = seg['safety_alert']
    sar, sag, sab = condition_color(alert_level)
    pdf.set_xy(sidebar_x, pdf.get_y() + 2)
    pdf.set_fill_color(sar, sag, sab)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 7)
    pdf.multi_cell(60, 4, f"SAFETY: {alert_text}", fill=True)

    heatmap_y = HEADER_H + 3
    if seg_heatmap_path and os.path.exists(seg_heatmap_path):
        pdf.image(seg_heatmap_path, x=10, y=heatmap_y, w=130, h=73)

    gap_y = heatmap_y + 75
    pdf.set_xy(10, gap_y)
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(60, 5, "CROWD DYNAMICS & ENVIRONMENT", align="L")
    pdf.ln(6)

    pdf.set_x(10)
    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(35, 5, f"Visibility Score: {seg['visibility']}%")
    pdf.cell(45, 5, f"Behavior: {seg['behavior']}")

    meter_x, meter_y = 92, gap_y + 6
    pdf.set_xy(meter_x, meter_y)
    pdf.set_font("helvetica", "B", 7)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(20, 4, "SENTIMENT:", align="L")

    bar_x, bar_w = meter_x + 18, 30
    pdf.set_fill_color(230, 230, 230)
    pdf.rect(bar_x, meter_y + 0.5, bar_w, 3, 'F')
    pdf.set_fill_color(34, 197, 94)
    indicator_pos = seg['sentiment'] * bar_w
    pdf.rect(bar_x, meter_y + 0.5, max(1.5, indicator_pos), 3, 'F')

    pdf.set_font("helvetica", "", 6)
    pdf.set_xy(bar_x, meter_y + 3.5)
    pdf.set_text_color(22, 101, 52)
    pdf.cell(bar_w / 2, 4, "Neutral", align="L")
    pdf.set_text_color(153, 27, 27)
    pdf.cell(bar_w / 2, 4, "Mob", align="R")

    thumb_y = 115
    pdf.set_y(thumb_y - 3)
    pdf.set_font("helvetica", "B", 8)
    pdf.set_text_color(10, 25, 75)
    pdf.cell(0, 5, "Processed Frames:", new_x="LMARGIN", new_y="NEXT")

    thumb_w = 37
    for i, (thumb_path, count, ts) in enumerate(thumb_paths_counts[:4]):
        if thumb_path and os.path.exists(thumb_path):
            tx = 10 + i * (thumb_w + 2)
            pdf.image(thumb_path, x=tx, y=thumb_y + 2, w=thumb_w, h=21)
            pdf.set_xy(tx, thumb_y + 24)
            pdf.set_font("helvetica", "", 6)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(thumb_w, 4, f"{ts}", align="C", new_x="RIGHT", new_y="TOP")
            pdf.set_xy(tx, thumb_y + 28)
            pdf.set_fill_color(50, 50, 50)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(thumb_w, 4, f"{count} ppl", align="C", fill=True, new_x="RIGHT", new_y="TOP")

    if seg.get('counts'):
        x_bar = 10
        y_bar = thumb_y + 36
        counts = seg['counts']
        fps_approx = 25
        seconds_axis = []
        second_vals = []
        for si in range(0, len(counts), fps_approx):
            chunk = counts[si:si + fps_approx]
            second_vals.append(np.mean(chunk))
            seconds_axis.append(si / fps_approx)
        plt.figure(figsize=(5.5, 1.8))
        plt.plot(seconds_axis, second_vals, color='#1e3a8a', linewidth=1.4)
        plt.fill_between(seconds_axis, second_vals, color='#1e3a8a', alpha=0.15)
        plt.title("Persons per Second (Segment)", fontsize=7, pad=4)
        plt.xlabel("Seconds into segment", fontsize=6)
        plt.ylabel("Count", fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.grid(True, alpha=0.2)
        plt.tight_layout(pad=0.3)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_bar:
            plt.savefig(tmp_bar.name, dpi=120)
            plt.close()
            pdf.image(tmp_bar.name, x=x_bar, y=y_bar, w=130, h=40)
            os.unlink(tmp_bar.name)


# ---------------------------------------------------------------------------
# Video processing (memory-efficient streaming)
# ---------------------------------------------------------------------------

def _emit_live_frame_ccn(job_id, frame, ccn_result, count, frame_idx, total_frames, fps):
    """Annotate frame with CCN density overlay and HUD, then publish to live stream buffer."""
    annotated = frame.copy()
    # Apply density heatmap overlay
    if ccn_result.get('heatmap') is not None and ccn_result.get('heat_alpha') is not None:
        heatmap = ccn_result['heatmap']
        heat_alpha = ccn_result['heat_alpha']
        h, w = annotated.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))
            heat_alpha = cv2.resize(heat_alpha, (w, h))
        alpha3 = np.stack([heat_alpha] * 3, axis=-1).astype(np.float32)
        annotated = (annotated.astype(np.float32) * (1 - alpha3) + heatmap.astype(np.float32) * alpha3).astype(np.uint8)

    h, w = annotated.shape[:2]
    bar_h = 36
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    elapsed = frame_idx / fps if fps > 0 else 0
    total_dur = total_frames / fps if fps > 0 else 0
    pct = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
    ts_text = f"{format_ts(elapsed)} / {format_ts(total_dur)}"

    cv2.putText(annotated, f"IRIS CROWD ANALYSIS | {count} Persons | {ts_text} | {pct}%",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

    bar_y = h - 4
    bar_width = int((frame_idx / total_frames) * w) if total_frames > 0 else 0
    cv2.rectangle(annotated, (0, bar_y), (bar_width, h), (0, 255, 200), -1)

    ret, buf = cv2.imencode('.jpg', annotated, _JPEG_QUALITY)
    if ret:
        with _live_frames_lock:
            _live_frames[job_id] = buf.tobytes()
            _live_frame_seq[job_id] = _live_frame_seq.get(job_id, 0) + 1


def _emit_live_frame(job_id, frame, boxes, count, frame_idx, total_frames, fps):
    """Annotate frame with detections and HUD, then publish to live stream buffer."""
    annotated = frame.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, "Head", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    h, w = annotated.shape[:2]
    # Dark HUD bar at bottom
    bar_h = 36
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    elapsed = frame_idx / fps if fps > 0 else 0
    total_dur = total_frames / fps if fps > 0 else 0
    pct = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
    ts_text = f"{format_ts(elapsed)} / {format_ts(total_dur)}"

    cv2.putText(annotated, f"IRIS CROWD ANALYSIS | {count} Persons | {ts_text} | {pct}%",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

    # Progress bar
    bar_y = h - 4
    bar_width = int((frame_idx / total_frames) * w) if total_frames > 0 else 0
    cv2.rectangle(annotated, (0, bar_y), (bar_width, h), (0, 255, 200), -1)

    ret, buf = cv2.imencode('.jpg', annotated, _JPEG_QUALITY)
    if ret:
        with _live_frames_lock:
            _live_frames[job_id] = buf.tobytes()
            _live_frame_seq[job_id] = _live_frame_seq.get(job_id, 0) + 1


def get_live_frame(job_id: str):
    """Get the latest JPEG frame for a job, or None."""
    with _live_frames_lock:
        return _live_frames.get(job_id), _live_frame_seq.get(job_id, 0)


def clear_live_frame(job_id: str):
    """Clean up live frame buffer when job completes."""
    with _live_frames_lock:
        _live_frames.pop(job_id, None)
        _live_frame_seq.pop(job_id, None)


def process_video(source_path, results_dir, model, device, segment_duration_sec=60, progress_cb=None, job_id=None):
    """Process video using CCN density model with YOLO-guided masking."""
    # Unpack hybrid model tuple
    if isinstance(model, tuple):
        ccn_model, yolo_model = model
    else:
        ccn_model, yolo_model = model, None
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_per_seg = int(fps * segment_duration_sec)
    infer_every = 12
    num_segs = math.ceil(total_frames / frames_per_seg) if total_frames > 0 else 1
    thumb_interval = max(1, frames_per_seg // 4)

    print(f"[CrowdReport] Processing {total_frames} frames, {num_segs} segments (CCN density model)")

    # Get Gemini road mask from a frame near 25% into the video (likely showing the scene well)
    sample_idx = min(total_frames - 1, int(total_frames * 0.25))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
    ret_s, sample_frame = cap.read()
    gemini_mask = None
    if ret_s:
        cache_key = os.path.basename(source_path)
        gemini_mask = _gemini_road_mask(sample_frame, cache_key=cache_key)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

    segments = []
    all_counts = []

    seg_idx = 0
    seg_counts = []
    seg_yolo_counts = []
    heatmap_acc = np.zeros((height, width), dtype=np.float32)
    mid_frame = None
    mid_density = None
    thumb_data = []
    last_count = 0
    last_yolo_count = 0
    last_density = None
    seg_start_sec = 0
    frame_in_seg = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % infer_every == 0:
            result = ccn_model.count(frame)
            # Run YOLO head detection once, reuse for both counting and suppression
            yolo_dets, yolo_conf = [], np.zeros((height, width), dtype=np.float32)
            if yolo_model is not None:
                yolo_dets, yolo_conf = _yolo_head_detect(yolo_model, frame)
            last_yolo_count = len(yolo_dets)
            # Suppress false positives — Gemini road mask + YOLO hybrid
            cleaned_density, cleaned_count = _suppress_non_crowd(
                frame, result['density_map'], None, gemini_road_mask=gemini_mask,
                _precomputed_yolo=(yolo_dets, yolo_conf))
            last_count = cleaned_count
            last_density = cleaned_density

            # Emit annotated frame for live streaming (with density overlay)
            if job_id:
                _emit_live_frame_ccn(job_id, frame, result, last_count, frame_idx, total_frames, fps)

        # Accumulate density map for heatmap
        if last_density is not None:
            h_d, w_d = last_density.shape[:2]
            if h_d != height or w_d != width:
                density_resized = cv2.resize(last_density, (width, height), interpolation=cv2.INTER_CUBIC)
            else:
                density_resized = last_density
            heatmap_acc += np.maximum(density_resized, 0)

        if frame_in_seg == frames_per_seg // 2:
            mid_frame = frame.copy()
            mid_density = last_density.copy() if last_density is not None else None

        if frame_in_seg % thumb_interval == 0 and len(thumb_data) < 4:
            ts = format_ts(seg_start_sec + frame_in_seg / fps)
            thumb_frame = frame.copy()
            # Overlay density heatmap on thumbnail
            if last_density is not None:
                thumb_frame = _overlay_density(thumb_frame, last_density)
            tpath = os.path.join(results_dir, f"tmp_thumb_s{seg_idx}_t{len(thumb_data)}.jpg")
            cv2.imwrite(tpath, cv2.resize(thumb_frame, (320, 180)))
            thumb_data.append((tpath, last_count, ts))
            del thumb_frame

        seg_counts.append(last_count)
        seg_yolo_counts.append(last_yolo_count)
        all_counts.append(last_count)
        frame_in_seg += 1

        # Report frame-level progress every ~2 seconds of video
        if progress_cb and frame_idx % max(1, int(fps * 2)) == 0:
            frame_pct = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
            progress_cb(seg_idx, num_segs, frame_pct)

        is_last = (frame_idx == total_frames - 1)
        if frame_in_seg >= frames_per_seg or is_last:
            seg_end_sec = seg_start_sec + frame_in_seg / fps

            base = mid_frame if mid_frame is not None else frame.copy()
            hmap_img = finalize_heatmap_density(heatmap_acc, base, width, height)
            hmap_path = os.path.join(results_dir, f"tmp_heatmap_s{seg_idx}.jpg")
            cv2.imwrite(hmap_path, hmap_img)
            del hmap_img

            # Save mid-frame for Gemini segment analysis
            mid_path = None
            if mid_frame is not None:
                mid_path = os.path.join(results_dir, f"tmp_mid_s{seg_idx}.jpg")
                cv2.imwrite(mid_path, mid_frame)

            yolo_avg = float(np.mean(seg_yolo_counts)) if seg_yolo_counts else 0
            segments.append({
                'idx': seg_idx + 1,
                'ts_start': format_ts(seg_start_sec),
                'ts_end': format_ts(seg_end_sec),
                'counts': seg_counts.copy(),
                'heatmap_path': hmap_path,
                'thumb_data': thumb_data.copy(),
                'mid_frame_path': mid_path,
                'yolo_head_count_avg': yolo_avg,
            })
            print(f"[CrowdReport] Segment {seg_idx+1}/{num_segs} | Avg:{np.mean(seg_counts):.1f} | Peak:{max(seg_counts)} | YOLO heads avg:{yolo_avg:.0f}")

            if progress_cb:
                progress_cb(seg_idx + 1, num_segs)

            seg_idx += 1
            seg_start_sec = seg_end_sec
            seg_counts, seg_yolo_counts, thumb_data = [], [], []
            heatmap_acc = np.zeros((height, width), dtype=np.float32)
            mid_frame, mid_density = None, None
            frame_in_seg = 0

    cap.release()

    fps_i = int(fps)
    per_second_avgs = [float(np.mean(all_counts[i:i+fps_i])) for i in range(0, len(all_counts), fps_i)]

    global_stats = {
        'source_url': source_path,
        'duration_processed': len(all_counts) / fps if fps > 0 else 0,
        'peak_count': max(all_counts) if all_counts else 0,
        'avg_count': float(np.mean(all_counts)) if all_counts else 0,
    }

    return segments, per_second_avgs, global_stats


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(source_path: str, output_pdf_path: str, results_dir: str, progress_cb=None, job_id=None):
    print(f"[CrowdReport] Starting report for {source_path}")
    os.makedirs(results_dir, exist_ok=True)

    model, device = _load_crowd_model()

    segments, per_second_avgs, global_stats = process_video(
        source_path, results_dir, model, device, progress_cb=progress_cb, job_id=job_id
    )
    phases = detect_phases(per_second_avgs)
    total_segs = len(segments)

    print("[CrowdReport] Generating AI event summary...")
    ai_summary = gemini_video_summary(global_stats, phases)

    print("[CrowdReport] Enriching segments with Gemini Vision analysis...")
    enriched = []
    for s in segments:
        counts = s['counts']
        avg = np.mean(counts) if counts else 0
        cong = congestion_score(avg)
        free = free_space_pct(avg)
        cond = condition_from_score(cong)
        fl = flow_direction(counts)
        predicted = predict_next_window(counts)
        pred_cond = condition_from_score(congestion_score(predicted))

        yolo_avg = s.get('yolo_head_count_avg', 0)
        seg_data = {'count': int(avg), 'condition': cond, 'congestion': cong,
                    'free_space': free, 'flow': fl, 'yolo_head_count': int(yolo_avg)}

        # Try Gemini Vision analysis on the mid-frame for accurate insights
        gemini_insights = None
        mid_path = s.get('mid_frame_path')
        if mid_path and os.path.exists(mid_path):
            mid_img = cv2.imread(mid_path)
            if mid_img is not None:
                gemini_insights = _gemini_segment_insights(mid_img, seg_data)

        if gemini_insights:
            narrative = gemini_insights.get('narrative', '') or gemini_segment_summary(seg_data)
            inss = gemini_insights.get('insights', []) or insights_from_segment(counts, cond, fl, predicted)
            safety_level = gemini_insights.get('safety_level', '')
            safety_note = gemini_insights.get('safety_note', '')
            level_map = {'critical': 'Critical', 'warning': 'High', 'caution': 'Medium', 'safe': 'Clear'}
            mapped_level = level_map.get(safety_level, cond)
            if safety_note:
                alert = (safety_note, mapped_level)
            else:
                alert = safety_alerts(cond, pred_cond)
            behavior = gemini_insights.get('behavior_description', f"Crowd trend: {gemini_insights.get('predicted_trend', 'stable')}")
            visibility = gemini_insights.get('visibility_pct', 90)
            sentiment_val = gemini_insights.get('sentiment', 0.1)
            # Override prediction with Gemini's forecast when available
            gem_pred = gemini_insights.get('predicted_count_next_60s')
            if gem_pred is not None:
                predicted = int(gem_pred)
                pred_cond = gemini_insights.get('predicted_condition_next_60s', pred_cond)
        else:
            narrative = gemini_segment_summary(seg_data)
            inss = insights_from_segment(counts, cond, fl, predicted)
            alert = safety_alerts(cond, pred_cond)
            behavior = "Calm pedestrian activity" if avg < 30 else "Normal crowd flow"
            visibility = 90
            sentiment_val = 0.1

        enriched.append({
            **s,
            'congestion_score': cong,
            'free_space': free,
            'condition': cond,
            'flow': fl,
            'predicted_count': predicted,
            'narrative': narrative,
            'insights': inss,
            'safety_alert': alert,
            'visibility': visibility,
            'sentiment': sentiment_val,
            'behavior': behavior,
        })
        print(f"[CrowdReport]   Segment {s['idx']}/{total_segs} | {cond.upper()} | Avg:{avg:.0f} | YOLO:{yolo_avg:.0f} | Gemini:{'yes' if gemini_insights else 'no'}")

    print("[CrowdReport] Generating PDF...")
    pdf = IrisPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    draw_cover_page(pdf, global_stats, phases, global_stats['duration_processed'], ai_summary)

    for seg in enriched:
        draw_segment_page(pdf, seg, seg['heatmap_path'], seg['thumb_data'], seg['idx'], total_segs)

    pdf.output(output_pdf_path)
    print(f"[CrowdReport] Report generated: {output_pdf_path}")

    for seg in enriched:
        for fp in [seg['heatmap_path']] + [t[0] for t in seg['thumb_data']] + [seg.get('mid_frame_path', '')]:
            try:
                if fp:
                    os.unlink(fp)
            except Exception:
                pass

    return output_pdf_path


# ---------------------------------------------------------------------------
# Job management (async background processing)
# ---------------------------------------------------------------------------

def create_job(filename: str, file_bytes: bytes) -> str:
    job_id = uuid.uuid4().hex
    job_dir = CROWD_REPORT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / filename
    with open(video_path, "wb") as f:
        f.write(file_bytes)

    return _start_job(job_id, filename, str(video_path))


def create_job_from_path(filename: str, file_path: str) -> str:
    """Create a job from a file already on disk (used by chunked upload)."""
    job_id = uuid.uuid4().hex
    job_dir = CROWD_REPORT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    target = job_dir / filename
    import shutil
    shutil.move(file_path, str(target))

    return _start_job(job_id, filename, str(target))


def _start_job(job_id: str, filename: str, video_path: str) -> str:
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "filename": filename,
            "status": "processing",
            "progress": 0,
            "total_segments": 0,
            "created_at": datetime.now().isoformat(),
            "video_path": video_path,
            "pdf_path": None,
            "error": None,
        }

    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    t.start()
    return job_id


def _run_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        video_path = job["video_path"]

    job_dir = CROWD_REPORT_DIR / job_id
    results_dir = str(job_dir / "tmp")
    os.makedirs(results_dir, exist_ok=True)
    pdf_path = str(job_dir / "crowd_report.pdf")

    def progress_cb(current_seg, total_segs, frame_pct=0):
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j:
                j["progress"] = current_seg
                j["total_segments"] = total_segs
                j["frame_pct"] = frame_pct

    try:
        build_report(video_path, pdf_path, results_dir, progress_cb=progress_cb, job_id=job_id)
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j:
                j["status"] = "completed"
                j["pdf_path"] = pdf_path
    except Exception as e:
        print(f"[CrowdReport] Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j:
                j["status"] = "failed"
                j["error"] = str(e)
    finally:
        clear_live_frame(job_id)


def get_status(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        return {
            "id": job["id"],
            "filename": job["filename"],
            "status": job["status"],
            "progress": job["progress"],
            "total_segments": job["total_segments"],
            "frame_pct": job.get("frame_pct", 0),
            "created_at": job["created_at"],
            "error": job["error"],
        }


def get_pdf_path(job_id: str) -> Optional[str]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job or job["status"] != "completed":
            return None
        return job.get("pdf_path")


def latest_job() -> Optional[dict]:
    with _jobs_lock:
        if not _jobs:
            return None
        latest = max(_jobs.values(), key=lambda j: j["created_at"])
        return {
            "id": latest["id"],
            "filename": latest["filename"],
            "status": latest["status"],
            "progress": latest["progress"],
            "total_segments": latest["total_segments"],
            "created_at": latest["created_at"],
            "error": latest["error"],
        }
