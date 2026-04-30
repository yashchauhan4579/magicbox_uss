#!/usr/bin/env python3
"""
Crowd Analysis Worker
Processes RTSP streams and sends crowd analysis data to the API endpoint.
Uses .pth density model for heatmaps + YOLO best_head.pt for head detection + Gemini for insights.
"""

import os
import sys
import cv2
import numpy as np
import requests
import logging
import time
import json
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy import ndimage

# Load .env file if present (for GEMINI_API_KEY etc.)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _, _v = _line.partition('=')
                _v = _v.strip().strip('"').strip("'")
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v

# Gemini analysis
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', os.getenv('GENAI_API_KEY', ''))
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False

def run_gemini_analysis(frame: np.ndarray, device_id: str, yolo_count: int = 0) -> Dict:
    """Run Gemini vision analysis using the IRIS original pipeline prompt."""
    if not GEMINI_AVAILABLE:
        return {}
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        model = genai.GenerativeModel('gemini-2.5-flash')
        now_str = datetime.now().strftime("%H:%M:%S")
        prompt = f"""You are a crowd safety analyst. {yolo_count} persons detected. Time: {now_str} IST.

Analyze and return ONLY a short JSON. Keep ALL text fields under 8 words max.

{{
  "crowd_movement": "<max 8 words>",
  "crowd_density": "LOW|MODERATE|HIGH|CRITICAL",
  "sentiment": "NEUTRAL|CAUTIOUS|AGITATED|MOB",
  "weapon_detected": "YES|NO",
  "fight_collision_injury": "YES|NO",
  "wrongful_activity": "YES|NO",
  "visibility_score": <0-100>,
  "predicted_count_next_segment": <integer>,
  "safety_precaution": "<max 8 words>",
  "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
  "behavior": "<max 8 words>"
}}

Be factual. Short answers only."""

        response = model.generate_content([prompt, img_pil])
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right != -1:
            text = text[left:right + 1]
        data = json.loads(text)
        data.setdefault("crowd_movement", "Stable movement.")
        data.setdefault("crowd_density", "LOW")
        data.setdefault("sentiment", "NEUTRAL")
        data.setdefault("weapon_detected", "NO")
        data.setdefault("fight_collision_injury", "NO")
        data.setdefault("wrongful_activity", "NO")
        data.setdefault("visibility_score", 90)
        data.setdefault("predicted_count_next_segment", yolo_count)
        data.setdefault("safety_precaution", "Continue standard monitoring.")
        data.setdefault("overall_risk", "LOW")
        data.setdefault("behavior", "Normal activity.")
        return data
    except Exception as e:
        logger.error(f"[{device_id}] Gemini error: {e}")
        return {}

# Add crowdanalysis to path if available (sibling directory)
_worker_dir = os.path.dirname(os.path.abspath(__file__))
CROWDANALYSIS_PATH = os.path.join(_worker_dir, 'crowdanalysis')
if os.path.exists(CROWDANALYSIS_PATH):
    sys.path.insert(0, CROWDANALYSIS_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:9010')
# Save heatmaps in user home directory to prevent file watcher overflow
USER_HOME = os.path.expanduser('~')
HEATMAP_SAVE_DIR = os.getenv('HEATMAP_SAVE_DIR', os.path.join(USER_HOME, 'heatmaps'))
os.makedirs(HEATMAP_SAVE_DIR, exist_ok=True)

# RTSP Authentication (default credentials)
RTSP_USERNAME = os.getenv('RTSP_USERNAME', 'admin')
RTSP_PASSWORD = os.getenv('RTSP_PASSWORD', 'admin123')

# Processing FPS (lower = less CPU, can handle more feeds)
# 0.2 FPS = one frame every 5 seconds (optimal for CPU)
PROCESSING_FPS = float(os.getenv('PROCESSING_FPS', '0.2'))

# Density model globals (.pth)
_model_net = None
_model_path = None
_net = None
_device = None
_img_transform = None
_model_loaded = False

# YOLO head detection globals (best_head.pt)
_yolo_model = None
_yolo_loaded = False
YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_head.pt')
YOLO_CONF = float(os.getenv('IRIS_CROWD_YOLO_CONF', '0.25'))
YOLO_IMGSZ = int(os.getenv('IRIS_CROWD_YOLO_IMGSZ', '960'))

# GPU inference lock — models are not thread-safe
import threading as _th
_inference_lock = _th.Lock()


def load_model():
    """Load the crowd counting model"""
    global _model_net, _model_path, _net, _device, _img_transform, _model_loaded
    
    try:
        from test_config import cfg
        
        _model_net = cfg.NET
        _model_path = cfg.MODEL_PATH
        
        # Resolve model path
        if not os.path.isabs(_model_path):
            _model_path = os.path.join(CROWDANALYSIS_PATH, _model_path)
        
        if not os.path.exists(_model_path):
            logger.error(f"Model file not found: {_model_path}")
            return False
        
        # Setup device
        if torch.cuda.is_available():
            _device = torch.device('cuda')
        else:
            _device = torch.device('cpu')
        
        # Load dataset config for transforms
        data_mode = cfg.DATASET
        if data_mode == 'SHHA':
            from datasets.SHHA.setting import cfg_data
        elif data_mode == 'SHHB':
            from datasets.SHHB.setting import cfg_data
        elif data_mode == 'QNRF':
            from datasets.QNRF.setting import cfg_data
        elif data_mode == 'UCF50':
            from datasets.UCF50.setting import cfg_data
        
        mean_std = cfg_data.MEAN_STD
        _img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        
        # Load model
        if 'LCM' in _model_net:
            from models.CC_LCM import CrowdCounter
        elif 'DM' in _model_net:
            from models.CC_DM import CrowdCounter
        else:
            logger.error(f"Unknown model type: {_model_net}")
            return False
        
        _net = CrowdCounter(cfg.GPU_ID, _model_net, pretrained=False)
        
        # Load weights
        if len(cfg.GPU_ID) == 1:
            _net.load_state_dict(torch.load(_model_path, map_location=_device))
        else:
            from collections import OrderedDict
            state_dict = torch.load(_model_path, map_location=_device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[0:3] + k[10:]
                new_state_dict[name] = v
            _net.load_state_dict(new_state_dict)
        
        _net.to(_device)
        _net.eval()
        
        logger.info(f"Model loaded successfully: {_model_net} on {_device}")
        _model_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"Could not load model: {e}", exc_info=True)
        return False


def load_yolo_model():
    """Load YOLO best_head.pt for head detection."""
    global _yolo_model, _yolo_loaded
    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.warning(f"YOLO model not found: {YOLO_MODEL_PATH}")
            return False
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_PATH, task="detect")
        if torch.cuda.is_available():
            _yolo_model.to('cuda')
        _yolo_loaded = True
        logger.info(f"YOLO head model loaded: {YOLO_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Could not load YOLO model: {e}", exc_info=True)
        return False


def run_yolo_detection(frame: np.ndarray) -> Tuple[int, list]:
    """Run YOLO head detection on a frame. Returns (head_count, list of bbox dicts)."""
    if not _yolo_loaded or _yolo_model is None:
        return 0, []
    try:
        results = _yolo_model.predict(
            frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
            max_det=300, verbose=False, classes=[0],
            half=torch.cuda.is_available(), device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        boxes = results[0].boxes
        count = len(boxes)
        bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            bboxes.append({'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'conf': round(conf, 2)})
        return count, bboxes
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
        return 0, []


def draw_yolo_boxes(frame: np.ndarray, bboxes: list) -> np.ndarray:
    """Draw YOLO detection boxes on frame."""
    out = frame.copy()
    for b in bboxes:
        color = (180, 180, 180)  # soft grey-white
        cv2.rectangle(out, (b['x1'], b['y1']), (b['x2'], b['y2']), color, 3)
    return out


def generate_heatmap(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heatmap and density map from image.
    Returns: (heatmap_image, density_map)
    """
    if not _model_loaded or _net is None:
        logger.warning("Model not loaded, using fallback")
        return _generate_heatmap_fallback(image)
    
    try:
        # Convert for model
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        img_tensor = _img_transform(img_pil)
        
        with torch.no_grad():
            img_tensor = Variable(img_tensor[None, :, :, :]).to(_device)
            pred_map = _net.test_forward(img_tensor)
        
        # Extract density map (same as original demo.py)
        density_map = pred_map.cpu().data.numpy()[0, 0, :, :]
        
        # Resize if needed (for DM models) - same as original
        if 'DM' in _model_net:
            from test_config import cfg
            density_map = cv2.resize(density_map, (density_map.shape[1]*8, density_map.shape[0]*8))
        
        # Generate heatmap using matplotlib (exactly like demo.py lines 594-599)
        # Use the density map directly, normalize based on its min/max
        min_val = float(np.min(density_map))
        max_val = float(np.max(density_map))
        
        logger.debug(f"Density map stats: min={min_val:.6f}, max={max_val:.6f}, shape={density_map.shape}")
        
        # Handle edge case where all values are the same
        if max_val <= min_val or max_val == 0:
            # Create a uniform heatmap (all zeros or constant)
            heatmap_final = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            logger.warning(f"Density map has no variation (min={min_val}, max={max_val})")
        else:
            # Normalize and create heatmap (exactly like original demo.py)
            # Ensure density map is non-negative (clip negative values to 0)
            density_map_clipped = np.maximum(density_map, 0)
            
            # Recalculate min/max after clipping
            min_val = float(np.min(density_map_clipped))
            max_val = float(np.max(density_map_clipped))
            
            if max_val > min_val:
                norm = plt.Normalize(vmin=min_val, vmax=max_val)
                cmap = plt.get_cmap('jet')
                heatmap_rgba = cmap(norm(density_map_clipped))
                heatmap_rgb = np.delete(heatmap_rgba, 3, 2)
                heatmap_bgr = cv2.cvtColor((heatmap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Resize to match original image (same as original demo.py line 602)
                heatmap_final = cv2.resize(heatmap_bgr, (image.shape[1], image.shape[0]))
            else:
                heatmap_final = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        return heatmap_final, density_map
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}", exc_info=True)
        return _generate_heatmap_fallback(image)


def _generate_heatmap_fallback(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback heatmap generation using HOG detector"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    small_img = cv2.resize(image, (640, 480))
    boxes, weights = hog.detectMultiScale(small_img, winStride=(8, 8))
    
    density_map = np.zeros((480, 640), dtype=np.float32)
    for (x, y, w, h) in boxes:
        center_x = x + w // 2
        center_y = y + h // 2
        radius = h // 2
        cv2.circle(density_map, (center_x, center_y), radius, (1,), -1)
    
    density_map = cv2.GaussianBlur(density_map, (31, 31), 0)
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    norm = plt.Normalize(vmin=np.min(density_map), vmax=np.max(density_map))
    cmap = plt.get_cmap('jet')
    heatmap_rgba = cmap(norm(density_map))
    heatmap_rgb = np.delete(heatmap_rgba, 3, 2)
    heatmap_bgr = cv2.cvtColor((heatmap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    heatmap_final = cv2.resize(heatmap_bgr, (image.shape[1], image.shape[0]))
    
    return heatmap_final, density_map


def calculate_crowd_metrics(density_map: np.ndarray, image_shape: Tuple[int, int], device_id: str) -> Dict:
    """Calculate comprehensive crowd metrics from density map"""
    logger.debug(f"[{device_id}] Calculating crowd metrics")
    
    # Basic statistics
    total_count = float(np.sum(density_map))
    max_density = float(np.max(density_map))
    mean_density = float(np.mean(density_map))
    std_density = float(np.std(density_map))
    
    # Calculate image area
    total_pixels = density_map.shape[0] * density_map.shape[1]
    
    # Define density thresholds
    low_threshold = mean_density + 0.5 * std_density
    medium_threshold = mean_density + 1.0 * std_density
    high_threshold = mean_density + 2.0 * std_density
    critical_threshold = mean_density + 3.0 * std_density
    
    # Find max density point
    max_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
    max_density_point = {
        'x': float(max_idx[1] / density_map.shape[1] * 100),  # Normalized to 0-100
        'y': float(max_idx[0] / density_map.shape[0] * 100),
        'density': max_density
    }
    
    # Hotspot zones
    hotspot_mask = density_map > high_threshold
    labeled_hotspots, num_hotspots = ndimage.label(hotspot_mask)
    
    hotspot_zones = []
    for i in range(1, num_hotspots + 1):
        hotspot_region = labeled_hotspots == i
        center = ndimage.center_of_mass(hotspot_region)
        radius = np.sqrt(np.sum(hotspot_region) / np.pi) * 2
        
        # Determine severity
        peak_density = float(np.max(density_map[hotspot_region]))
        if peak_density > critical_threshold:
            severity = 'RED'
        elif peak_density > high_threshold:
            severity = 'ORANGE'
        else:
            severity = 'YELLOW'
        
        hotspot_zones.append({
            'x': float(center[1] / density_map.shape[1] * 100),
            'y': float(center[0] / density_map.shape[0] * 100),
            'radius': float(radius / density_map.shape[1] * 100),
            'severity': severity
        })
    
    # Determine density level
    if max_density > critical_threshold:
        density_level = 'CRITICAL'
    elif max_density > high_threshold:
        density_level = 'HIGH'
    elif max_density > medium_threshold:
        density_level = 'MEDIUM'
    else:
        density_level = 'LOW'
    
    # Determine hotspot severity (overall)
    if max_density > critical_threshold:
        hotspot_severity = 'RED'
    elif max_density > high_threshold:
        hotspot_severity = 'ORANGE'
    elif max_density > medium_threshold:
        hotspot_severity = 'YELLOW'
    else:
        hotspot_severity = 'GREEN'
    
    # Calculate occupancy and free space
    occupancy_rate = float(np.sum(density_map > mean_density) / total_pixels)
    free_space = float((1.0 - occupancy_rate) * 100)
    
    # Congestion level (0-10 scale)
    congestion_level = int(min(10, max(0, (max_density / critical_threshold) * 10))) if critical_threshold > 0 else 0
    
    # Normalize density value (0-1)
    density_value = float(min(1.0, max_density / (critical_threshold * 2))) if critical_threshold > 0 else 0.0
    
    return {
        'peopleCount': int(total_count + 0.5),
        'densityValue': density_value,
        'densityLevel': density_level,
        'hotspotSeverity': hotspot_severity,
        'hotspotZones': hotspot_zones,
        'maxDensityPoint': max_density_point,
        'freeSpace': free_space,
        'congestionLevel': congestion_level,
        'occupancyRate': occupancy_rate,
    }


def overlay_heatmap_on_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay heatmap on original frame with transparency (same as original demo.py)"""
    # Ensure both images are the same size
    if frame.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    
    # Blend heatmap with original frame (same as original demo.py line 606)
    # alpha controls heatmap opacity, (1-alpha) controls frame opacity
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return overlay


def save_heatmap_image(frame: np.ndarray, heatmap: np.ndarray, device_id: str, timestamp: str) -> str:
    """Save heatmap overlaid on original frame and return URL path"""
    # Overlay heatmap on original frame
    overlay = overlay_heatmap_on_frame(frame, heatmap, alpha=0.5)
    
    filename = f"{device_id}_{timestamp}.jpg"
    filepath = os.path.join(HEATMAP_SAVE_DIR, filename)
    cv2.imwrite(filepath, overlay)
    
    # Return URL path (assuming heatmaps are served statically)
    # In production, you might upload to S3 or similar
    return f"/heatmaps/{filename}"


def _compute_congestion(yolo_count: int, density_count: int) -> Tuple[str, int]:
    """Determine density level and congestion score. Weighted blend: 25% density, 75% YOLO."""
    count = int(density_count * 0.25 + yolo_count * 0.75) if density_count > 0 else yolo_count
    if count >= 50:
        return 'CRITICAL', 10
    elif count >= 30:
        return 'HIGH', min(10, 6 + (count - 30) // 5)
    elif count >= 15:
        return 'MEDIUM', min(10, 3 + (count - 15) // 5)
    elif count >= 5:
        return 'LOW', min(10, 1 + count // 3)
    elif count >= 1:
        return 'LOW', 1
    else:
        return 'LOW', 0


def process_frame_gpu(frame: np.ndarray, device_id: str) -> Optional[Dict]:
    """GPU inference only: heatmap + YOLO detection + overlay save. No Gemini."""
    try:
        with _inference_lock:
            heatmap, density_map = generate_heatmap(frame)
            yolo_count, yolo_bboxes = run_yolo_detection(frame)

        metrics = calculate_crowd_metrics(density_map, frame.shape[:2], device_id)
        people_count = yolo_count if _yolo_loaded else metrics['peopleCount']
        density_level, congestion = _compute_congestion(yolo_count, metrics['peopleCount'])

        overlay = overlay_heatmap_on_frame(frame, heatmap, alpha=0.5)
        if yolo_bboxes:
            overlay = draw_yolo_boxes(overlay, yolo_bboxes)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{device_id}_{timestamp}.jpg"
        filepath = os.path.join(HEATMAP_SAVE_DIR, filename)
        cv2.imwrite(filepath, overlay)
        heatmap_url = f"/heatmaps/{filename}"

        return {
            'device_id': device_id,
            'frame': frame,
            'people_count': people_count,
            'yolo_count': yolo_count,
            'yolo_bboxes': yolo_bboxes,
            'metrics': metrics,
            'density_level': density_level,
            'congestion': congestion,
            'heatmap_url': heatmap_url,
        }
    except Exception as e:
        logger.error(f"[{device_id}] GPU inference error: {e}", exc_info=True)
        return None


def _enrich_with_gemini(gpu_result: Dict) -> Optional[Dict]:
    """Run Gemini API call and build final analysis_data. Thread-safe (no GPU)."""
    device_id = gpu_result['device_id']
    frame = gpu_result['frame']
    people_count = gpu_result['people_count']
    yolo_count = gpu_result['yolo_count']
    yolo_bboxes = gpu_result['yolo_bboxes']
    metrics = gpu_result['metrics']
    density_level = gpu_result['density_level']
    congestion = gpu_result['congestion']
    heatmap_url = gpu_result['heatmap_url']

    gemini = {}
    if GEMINI_AVAILABLE:
        try:
            gemini = run_gemini_analysis(frame, device_id, yolo_count=people_count)
            if gemini:
                logger.info(f"[{device_id}] Gemini: risk={gemini.get('overall_risk')}, "
                            f"sentiment={gemini.get('sentiment')}, behavior={gemini.get('behavior','')[:60]}")
        except Exception as e:
            logger.warning(f"[{device_id}] Gemini skipped: {e}")

    return {
        'deviceId': device_id,
        'peopleCount': people_count,
        'yoloCount': yolo_count,
        'densityCount': metrics['peopleCount'],
        'densityValue': metrics['densityValue'],
        'densityLevel': density_level,
        'movementType': 'FLOWING',
        'freeSpace': metrics['freeSpace'],
        'congestionLevel': congestion,
        'occupancyRate': metrics['occupancyRate'],
        'hotspotSeverity': metrics['hotspotSeverity'],
        'hotspotZones': metrics['hotspotZones'],
        'maxDensityPoint': metrics['maxDensityPoint'],
        'heatmapImageUrl': heatmap_url,
        'crowd_movement': gemini.get('crowd_movement', 'Stable movement.'),
        'crowd_density': gemini.get('crowd_density', density_level),
        'sentiment': gemini.get('sentiment', 'NEUTRAL'),
        'weapon_detected': gemini.get('weapon_detected', 'NO'),
        'fight_collision_injury': gemini.get('fight_collision_injury', 'NO'),
        'wrongful_activity': gemini.get('wrongful_activity', 'NO'),
        'visibility_score': gemini.get('visibility_score', 90),
        'predicted_count': gemini.get('predicted_count_next_segment', people_count),
        'safety_precaution': gemini.get('safety_precaution', 'Continue standard monitoring.'),
        'overall_risk': gemini.get('overall_risk', 'LOW'),
        'behavior': gemini.get('behavior', 'Normal activity.'),
        'modelType': 'hybrid',
        'confidence': 0.95 if gemini else 0.9,
        'timestamp': datetime.now().isoformat(),
        'yoloBboxes': len(yolo_bboxes),
    }


def process_frame(frame: np.ndarray, device_id: str) -> Optional[Dict]:
    """Legacy wrapper: GPU inference + Gemini in one call."""
    gpu_result = process_frame_gpu(frame, device_id)
    if not gpu_result:
        return None
    return _enrich_with_gemini(gpu_result)


def send_analysis_to_api(analysis_data: Dict) -> bool:
    """Send analysis data to API endpoint"""
    try:
        url = f"{API_BASE_URL}/api/crowd/analysis"
        response = requests.post(url, json=analysis_data, timeout=10)
        
        if response.status_code in [200, 201]:
            logger.info(f"[{analysis_data['deviceId']}] Successfully sent analysis data")
            return True
        else:
            logger.error(f"[{analysis_data['deviceId']}] API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"[{analysis_data['deviceId']}] Error sending to API: {e}", exc_info=True)
        return False


def capture_single_frame(device_id: str, rtsp_url: str, timeout: int = 6) -> Optional[np.ndarray]:
    """Capture a single frame from RTSP using ffmpeg subprocess (more reliable than OpenCV)."""
    import subprocess
    tmp_path = f"/tmp/crowd_capture_{device_id}.jpg"
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-timeout", "4000000",
            "-i", rtsp_url,
            "-ss", "1.5",
            "-frames:v", "1",
            "-q:v", "2",
            tmp_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if proc.returncode != 0 or not os.path.exists(tmp_path):
            stderr = proc.stderr.decode(errors='ignore')[:200] if proc.stderr else ""
            logger.warning(f"[{device_id}] ffmpeg capture failed: {stderr}")
            return None
        frame = cv2.imread(tmp_path)
        os.unlink(tmp_path)
        if frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            if h >= 10 and w >= 10:
                return frame
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"[{device_id}] Capture timed out after {timeout}s")
        return None
    except Exception as e:
        logger.warning(f"[{device_id}] Capture error: {e}")
        return None


def _capture_task(task: dict, fail_counts: dict) -> tuple:
    """Capture a single frame for a task. Returns (device_id, frame_or_None)."""
    device_id = task['id']
    rtsp_url = task['rtspUrl']
    fails = fail_counts.get(device_id, 0)
    if fails >= 5:
        fail_counts[device_id] = 0
        return (device_id, None, True)  # skipped
    frame = capture_single_frame(device_id, rtsp_url)
    return (device_id, frame, False)


def _build_analysis_data_no_gemini(gpu_result: Dict) -> Optional[Dict]:
    """Build analysis_data without Gemini — uses defaults for all Gemini fields."""
    device_id = gpu_result['device_id']
    people_count = gpu_result['people_count']
    yolo_count = gpu_result['yolo_count']
    yolo_bboxes = gpu_result['yolo_bboxes']
    metrics = gpu_result['metrics']
    density_level = gpu_result['density_level']
    congestion = gpu_result['congestion']
    heatmap_url = gpu_result['heatmap_url']

    return {
        'deviceId': device_id,
        'peopleCount': people_count,
        'yoloCount': yolo_count,
        'densityCount': metrics['peopleCount'],
        'densityValue': metrics['densityValue'],
        'densityLevel': density_level,
        'movementType': 'FLOWING',
        'freeSpace': metrics['freeSpace'],
        'congestionLevel': congestion,
        'occupancyRate': metrics['occupancyRate'],
        'hotspotSeverity': metrics['hotspotSeverity'],
        'hotspotZones': metrics['hotspotZones'],
        'maxDensityPoint': metrics['maxDensityPoint'],
        'heatmapImageUrl': heatmap_url,
        'crowd_movement': 'Stable movement.',
        'crowd_density': density_level,
        'sentiment': 'NEUTRAL',
        'weapon_detected': 'NO',
        'fight_collision_injury': 'NO',
        'wrongful_activity': 'NO',
        'visibility_score': 90,
        'predicted_count': people_count,
        'safety_precaution': 'Continue standard monitoring.',
        'overall_risk': 'LOW',
        'behavior': 'Normal activity.',
        'modelType': 'hybrid',
        'confidence': 0.9,
        'timestamp': datetime.now().isoformat(),
        'yoloBboxes': len(yolo_bboxes),
    }


def process_all_cameras_sequential(tasks: list):
    """Process cameras with alternating groups + parallel capture + sequential GPU inference.

    Splits cameras into 2 groups, processes one group per cycle. This halves GPU time
    per cycle (~12s instead of ~20s), giving ~25 frames per camera per 5 minutes.
    Gemini runs every 6th cycle (every 3rd cycle per group) to keep capture rate high.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    fail_counts = {}
    cycle_count = 0
    gemini_interval = 6  # Gemini every 6th cycle = every 3rd time each group runs
    max_capture_workers = min(8, len(tasks))

    # Split into 2 alternating groups
    group_a = tasks[::2]   # even indices
    group_b = tasks[1::2]  # odd indices
    groups = [group_a, group_b]
    logger.info(f"Alternating groups: A={len(group_a)} cams, B={len(group_b)} cams (Gemini every {gemini_interval} cycles)")

    while True:
        cycle_start = time.time()
        processed = 0
        cycle_count += 1
        run_gemini = (cycle_count % gemini_interval == 0)

        # Alternate between groups each cycle
        current_group = groups[(cycle_count - 1) % 2]
        group_label = 'A' if (cycle_count - 1) % 2 == 0 else 'B'

        # Phase 1: Capture group frames in parallel
        captured_frames = {}
        capture_start = time.time()
        with ThreadPoolExecutor(max_workers=max_capture_workers) as pool:
            futures = {pool.submit(_capture_task, task, fail_counts): task for task in current_group}
            for future in as_completed(futures):
                try:
                    device_id, frame, skipped = future.result()
                    if skipped:
                        continue
                    if frame is not None:
                        captured_frames[device_id] = frame
                        fail_counts[device_id] = 0
                    else:
                        fail_counts[device_id] = fail_counts.get(device_id, 0) + 1
                except Exception as e:
                    task = futures[future]
                    logger.error(f"[{task['id']}] Capture error: {e}")
                    fail_counts[task['id']] = fail_counts.get(task['id'], 0) + 1

        capture_elapsed = time.time() - capture_start
        logger.info(f"[Group {group_label}] Captured {len(captured_frames)}/{len(current_group)} frames in {capture_elapsed:.1f}s")

        # Phase 2: GPU inference sequentially
        gpu_results = []
        infer_start = time.time()
        for device_id, frame in captured_frames.items():
            try:
                gpu_result = process_frame_gpu(frame, device_id)
                if gpu_result:
                    gpu_results.append(gpu_result)
            except Exception as e:
                logger.error(f"[{device_id}] Inference error: {e}")
        infer_elapsed = time.time() - infer_start
        logger.info(f"[Group {group_label}] GPU inference: {len(gpu_results)} frames in {infer_elapsed:.1f}s")

        # Phase 3: Gemini every Nth cycle, skip on others
        if not gpu_results:
            pass
        elif run_gemini:
            gemini_start = time.time()
            with ThreadPoolExecutor(max_workers=min(6, len(gpu_results))) as pool:
                futures = {pool.submit(_enrich_with_gemini, r): r for r in gpu_results}
                for future in as_completed(futures):
                    try:
                        analysis_data = future.result()
                        if analysis_data:
                            send_analysis_to_api(analysis_data)
                            processed += 1
                    except Exception as e:
                        r = futures[future]
                        logger.error(f"[{r['device_id']}] Gemini/send error: {e}")
            gemini_elapsed = time.time() - gemini_start
            logger.info(f"[Group {group_label}] Gemini enrichment: {processed} cameras in {gemini_elapsed:.1f}s")
        else:
            for gpu_result in gpu_results:
                try:
                    analysis_data = _build_analysis_data_no_gemini(gpu_result)
                    if analysis_data:
                        send_analysis_to_api(analysis_data)
                        processed += 1
                except Exception as e:
                    logger.error(f"[{gpu_result['device_id']}] Send error: {e}")

        elapsed = time.time() - cycle_start
        logger.info(f"Cycle done: {processed}/{len(current_group)} [Group {group_label}] in {elapsed:.1f}s (gemini={'yes' if run_gemini else 'skip'})")

        time.sleep(1.0)


def add_rtsp_auth(rtsp_url: str) -> str:
    """Add RTSP authentication to URL if not already present"""
    from urllib.parse import urlparse, urlunparse
    
    # Check if URL already has authentication (contains @ after protocol)
    if '@' in rtsp_url.split('://', 1)[1] if '://' in rtsp_url else '':
        # Already has auth, return as-is
        return rtsp_url
    
    try:
        # Parse the URL
        parsed = urlparse(rtsp_url)
        
        # Reconstruct with authentication
        # Format: rtsp://username:password@host:port/path
        netloc = f"{RTSP_USERNAME}:{RTSP_PASSWORD}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        
        # Reconstruct URL
        new_parsed = parsed._replace(netloc=netloc)
        return urlunparse(new_parsed)
    except Exception as e:
        logger.warning(f"Failed to add RTSP auth to {rtsp_url}: {e}, using original URL")
        return rtsp_url


def fetch_worker_config(worker_id: Optional[str] = None) -> Dict:
    """Fetch worker configuration from API"""
    try:
        url = f"{API_BASE_URL}/api/crowd-worker/config"
        if worker_id:
            url += f"?workerId={worker_id}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch worker config: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error fetching worker config: {e}", exc_info=True)
        return {}


def _cleanup_old_heatmaps():
    """Delete heatmap files older than 75 minutes to allow hourly report generation."""
    cutoff = time.time() - 4500
    try:
        for f in Path(HEATMAP_SAVE_DIR).glob("*.jpg"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception:
        pass


def _heatmap_cleanup_loop():
    """Background thread to clean up old heatmaps every 60s."""
    while True:
        time.sleep(60)
        _cleanup_old_heatmaps()


def main():
    """Main worker loop"""
    logger.info("Starting Crowd Analysis Worker")

    # Load density model (.pth)
    if not load_model():
        logger.error("Failed to load density model. Exiting.")
        return

    # Load YOLO head detection model
    if not load_yolo_model():
        logger.warning("YOLO head model not loaded — will use density count only")
    
    # Fetch worker config
    worker_id = os.getenv('WORKER_ID')
    config = fetch_worker_config(worker_id)
    
    if not config or 'tasks' not in config:
        logger.error("No tasks found in worker config")
        return
    
    tasks = config['tasks']
    logger.info(f"Found {len(tasks)} tasks")
    
    # Use all cameras (up to 15)
    tasks = tasks[:20]
    logger.info(f"Processing {len(tasks)} devices")
    
    # Start heatmap cleanup thread
    import threading
    threading.Thread(target=_heatmap_cleanup_loop, daemon=True).start()

    # Prepare task list with auth
    valid_tasks = []
    for task in tasks:
        device_id = task['id']
        rtsp_url = task.get('rtspUrl')
        if not rtsp_url:
            logger.warning(f"[{device_id}] No RTSP URL found, skipping")
            continue
        rtsp_url = add_rtsp_auth(rtsp_url)
        valid_tasks.append({'id': device_id, 'rtspUrl': rtsp_url})

    # Sequential processing — one camera at a time, zero GPU contention
    try:
        process_all_cameras_sequential(valid_tasks)
    except KeyboardInterrupt:
        logger.info("Shutting down worker")


if __name__ == '__main__':
    main()

