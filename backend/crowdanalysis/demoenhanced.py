# type: ignore
import os
import cv2
import math
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
import argparse
import time
from datetime import datetime
import requests
import base64
import json
import logging
import yaml
import threading
from scipy import ndimage

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

from misc.utils import *
from test_config import cfg

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration is now loaded from config.yaml

''' prepare model config '''
model_net = cfg.NET
model_path = cfg.MODEL_PATH

cfg_GPU_ID = cfg.GPU_ID

# Check if CUDA is available, otherwise use CPU
if torch.cuda.is_available():
    logger.info(f"CUDA is available. Using GPU device {cfg_GPU_ID[0]}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(cfg_GPU_ID[0])}")
    torch.cuda.set_device(cfg_GPU_ID[0])
    device = torch.device('cuda')
else:
    logger.info("CUDA is not available. Using CPU")
    device = torch.device('cpu')

logger.info(f"Using device: {device}")
torch.backends.cudnn.benchmark = True


''' prepare data config '''
data_mode = cfg.DATASET
if data_mode == 'SHHA':
    from datasets.SHHA.setting import cfg_data
elif data_mode == 'SHHB':
    from datasets.SHHB.setting import cfg_data
elif data_mode == 'QNRF':
    from datasets.QNRF.setting import cfg_data
elif data_mode == 'UCF50':
    from datasets.UCF50.setting import cfg_data
    val_index = cfg_data.VAL_INDEX
    
mean_std = cfg_data.MEAN_STD
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])


def load_gpus_to_gpu(model, model_path):
    ''' convert multi-gpu trained model to single model '''
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    # create new OrderedDict that does not contain 'module.'
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[0:3] + k[10:]  # remove 'module.'
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
    return model


def send_metric_to_api(count, overlay_img, camera_id, api_endpoint, api_token, crowd_metrics=None):
    try:
        logger.info(f"[{camera_id}] Preparing to send metric data for count: {count}")

        # Convert overlay image to base64
        _, buffer = cv2.imencode('.jpg', overlay_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.debug(f"[{camera_id}] Image converted to base64 successfully")

        # Determine density level based on count
        # These thresholds can be adjusted based on your requirements
        if count < 10:
            density_level = "low"
        elif count < 50:
            density_level = "medium"
        else:
            density_level = "high"
            
        logger.info(f"[{camera_id}] Density level determined: {density_level} (count: {count})")
        count = int(count * 1.8)
        logger.info(f"[{camera_id}] Count adjusted by 1.8x multiplier: {count}")
        
        # Prepare base metadata
        base_metadata = {
            "density_level": density_level,
            "flow": "static"
        }
        
        # Add comprehensive crowd metrics if available
        if crowd_metrics:
            # Console log all enhanced metrics before sending to API
            logger.info(f"[{camera_id}] ========== ENHANCED CROWD METRICS ==========")
            logger.info(f"[{camera_id}] Basic Metrics:")
            logger.info(f"[{camera_id}]   - Total Count: {crowd_metrics.get('total_count', 0):.1f}")
            logger.info(f"[{camera_id}]   - Peak Density: {crowd_metrics.get('peak_density', 0):.2f}")
            logger.info(f"[{camera_id}]   - Average Density: {crowd_metrics.get('average_density', 0):.2f}")
            logger.info(f"[{camera_id}]   - Density Std Dev: {crowd_metrics.get('density_std', 0):.2f}")
            
            logger.info(f"[{camera_id}] Critical Density Metrics:")
            logger.info(f"[{camera_id}]   - Critical Area %: {crowd_metrics.get('critical_density_area_pct', 0):.2f}%")
            logger.info(f"[{camera_id}]   - Critical Pixels: {crowd_metrics.get('critical_density_pixels', 0)}")
            logger.info(f"[{camera_id}]   - Critical Threshold: {crowd_metrics.get('critical_threshold', 0):.2f}")
            
            logger.info(f"[{camera_id}] Hotspot Metrics:")
            logger.info(f"[{camera_id}]   - Number of Hotspots: {crowd_metrics.get('num_hotspots', 0)}")
            logger.info(f"[{camera_id}]   - Hotspot Area %: {crowd_metrics.get('hotspot_area_pct', 0):.2f}%")
            logger.info(f"[{camera_id}]   - Largest Hotspot Size: {crowd_metrics.get('largest_hotspot_size', 0)} pixels")
            logger.info(f"[{camera_id}]   - Average Hotspot Size: {crowd_metrics.get('average_hotspot_size', 0):.1f} pixels")
            
            hotspot_peaks = crowd_metrics.get('hotspot_peak_densities', [])
            if hotspot_peaks:
                logger.info(f"[{camera_id}]   - Hotspot Peak Densities: {[f'{peak:.2f}' for peak in hotspot_peaks[:5]]}")  # Show first 5
            
            logger.info(f"[{camera_id}] Spatial Distribution:")
            logger.info(f"[{camera_id}]   - Crowd Coverage %: {crowd_metrics.get('crowd_coverage_pct', 0):.2f}%")
            logger.info(f"[{camera_id}]   - Crowd Pattern: {crowd_metrics.get('crowd_pattern', 'unknown')}")
            logger.info(f"[{camera_id}]   - Density Coefficient of Variation: {crowd_metrics.get('density_coefficient_variation', 0):.2f}")
            
            density_zones = crowd_metrics.get('density_zones_pct', {})
            logger.info(f"[{camera_id}] Density Zones:")
            logger.info(f"[{camera_id}]   - No Crowd: {density_zones.get('no_crowd', 0):.1f}%")
            logger.info(f"[{camera_id}]   - Low Density: {density_zones.get('low_density', 0):.1f}%")
            logger.info(f"[{camera_id}]   - Medium Density: {density_zones.get('medium_density', 0):.1f}%")
            logger.info(f"[{camera_id}]   - High Density: {density_zones.get('high_density', 0):.1f}%")
            logger.info(f"[{camera_id}]   - Critical Density: {density_zones.get('critical_density', 0):.1f}%")
            
            percentiles = crowd_metrics.get('density_percentiles', {})
            logger.info(f"[{camera_id}] Density Percentiles:")
            logger.info(f"[{camera_id}]   - P25: {percentiles.get('p25', 0):.2f}, P50: {percentiles.get('p50', 0):.2f}, P75: {percentiles.get('p75', 0):.2f}")
            logger.info(f"[{camera_id}]   - P90: {percentiles.get('p90', 0):.2f}, P95: {percentiles.get('p95', 0):.2f}")
            
            thresholds = crowd_metrics.get('thresholds', {})
            logger.info(f"[{camera_id}] Analysis Thresholds:")
            logger.info(f"[{camera_id}]   - Low: {thresholds.get('low', 0):.2f}, Medium: {thresholds.get('medium', 0):.2f}")
            logger.info(f"[{camera_id}]   - High: {thresholds.get('high', 0):.2f}, Critical: {thresholds.get('critical', 0):.2f}")
            logger.info(f"[{camera_id}] ===============================================")
            
            base_metadata.update({
                # Critical density metrics
                "critical_density_area_pct": crowd_metrics.get('critical_density_area_pct', 0),
                "critical_density_pixels": crowd_metrics.get('critical_density_pixels', 0),
                
                # Hotspot metrics
                "num_hotspots": crowd_metrics.get('num_hotspots', 0),
                "hotspot_area_pct": crowd_metrics.get('hotspot_area_pct', 0),
                "largest_hotspot_size": crowd_metrics.get('largest_hotspot_size', 0),
                "average_hotspot_size": crowd_metrics.get('average_hotspot_size', 0),
                
                # Density analysis
                "peak_density": crowd_metrics.get('peak_density', 0),
                "average_density": crowd_metrics.get('average_density', 0),
                "crowd_pattern": crowd_metrics.get('crowd_pattern', 'unknown'),
                "density_coefficient_variation": crowd_metrics.get('density_coefficient_variation', 0),
                
                # Coverage metrics
                "crowd_coverage_pct": crowd_metrics.get('crowd_coverage_pct', 0),
                
                # Zone analysis
                "density_zones_pct": crowd_metrics.get('density_zones_pct', {}),
                
                # Density percentiles
                "density_percentiles": crowd_metrics.get('density_percentiles', {}),
                
                # Thresholds used for analysis
                "analysis_thresholds": crowd_metrics.get('thresholds', {})
            })
            
            logger.info(f"[{camera_id}] Enhanced metrics prepared for API transmission")
        
        # Prepare the payload
        payload = {
            "cameraId": camera_id,
            "metricType": "crowd_count",
            "value_numeric": round(float(count), 2),
            "unit": "people",
            "timestamp": datetime.now().isoformat(),
            "imageData": img_base64,
            "metadata": base_metadata
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        payload = convert_numpy_types(payload)
        logger.debug(f"[{camera_id}] Payload converted to JSON-serializable format")
        
        # Log the complete API payload structure (without image data for readability)
        logger.info(f"[{camera_id}] ========== API PAYLOAD STRUCTURE ==========")
        logger.info(f"[{camera_id}] Camera ID: {payload['cameraId']}")
        logger.info(f"[{camera_id}] Metric Type: {payload['metricType']}")
        logger.info(f"[{camera_id}] Count Value: {payload['value_numeric']}")
        logger.info(f"[{camera_id}] Unit: {payload['unit']}")
        logger.info(f"[{camera_id}] Timestamp: {payload['timestamp']}")
        logger.info(f"[{camera_id}] Image Data: [BASE64_IMAGE_DATA - {len(img_base64)} chars]")
        logger.info(f"[{camera_id}] Metadata Keys: {list(base_metadata.keys())}")
        logger.info(f"[{camera_id}] Full Metadata:")
        for key, value in base_metadata.items():
            if isinstance(value, dict):
                # Limit dict logging to avoid large data dumps
                dict_str = json.dumps(value, indent=2)
                if len(dict_str) > 500:  # Truncate large dicts
                    logger.info(f"[{camera_id}]   {key}: [LARGE_DICT - {len(dict_str)} chars - keys: {list(value.keys())}]")
                else:
                    logger.info(f"[{camera_id}]   {key}: {dict_str}")
            elif isinstance(value, str) and len(value) > 100:  # Truncate long strings
                logger.info(f"[{camera_id}]   {key}: [LONG_STRING - {len(value)} chars]")
            else:
                logger.info(f"[{camera_id}]   {key}: {value}")
        logger.info(f"[{camera_id}] ================================================")
        
        # Send POST request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}'
        }
        
        logger.info(f"[{camera_id}] Sending API request to {api_endpoint}")
        response = requests.post(api_endpoint,
                               headers=headers,
                               json=payload)
        
        if response.status_code in [200, 202]:
            logger.info(f"[{camera_id}] Successfully sent metric data. Count: {count}, Status: {response.status_code}")
            if response.status_code == 202 and response.text:
                response_text = response.text
                if len(response_text) > 200:
                    logger.info(f"[{camera_id}] Response (202 Accepted): [LARGE_RESPONSE - {len(response_text)} chars]")
                else:
                    logger.info(f"[{camera_id}] Response (202 Accepted): {response_text}")
        else:
            logger.error(f"[{camera_id}] Failed to send metric data. Status code: {response.status_code}")
            if response.text:
                response_text = response.text
                if len(response_text) > 200:
                    logger.error(f"[{camera_id}] Response: [LARGE_ERROR_RESPONSE - {len(response_text)} chars]")
                else:
                    logger.error(f"[{camera_id}] Response: {response_text}")
            
    except Exception as e:
        error_str = str(e)
        if len(error_str) > 300:
            logger.error(f"[{camera_id}] Error sending metric data: [LARGE_ERROR - {len(error_str)} chars] - {error_str[:300]}...")
        else:
            logger.error(f"[{camera_id}] Error sending metric data: {error_str}")


def send_crowd_analysis_to_api(raw_img, annotated_img, camera_id, api_token, crowd_metrics=None):
    try:
        api_endpoint = 'http://localhost:3000/api/v1/crowd'
        logger.info(f"[{camera_id}] Preparing to send crowd analysis data to {api_endpoint}")

        # Convert raw image to base64 with data URI scheme
        _, raw_buffer = cv2.imencode('.jpg', raw_img)
        raw_img_base64 = base64.b64encode(raw_buffer).decode('utf-8')
        raw_img_payload = f"data:image/jpeg;base64,{raw_img_base64}"
        logger.debug(f"[{camera_id}] Raw image converted to base64 successfully")

        # Convert annotated image to base64 with data URI scheme
        _, annotated_buffer = cv2.imencode('.jpg', annotated_img)
        annotated_img_base64 = base64.b64encode(annotated_buffer).decode('utf-8')
        annotated_img_payload = f"data:image/jpeg;base64,{annotated_img_base64}"
        logger.debug(f"[{camera_id}] Annotated image converted to base64 successfully")

        # Calculate stampede risk score and alert level based on metrics
        stampede_risk_score = 0
        alert_level = "Low"
        alert_threshold_crossed = False
        risk_profile = "NORMAL"
        title = "Normal Crowd Conditions"
        description = "Standard crowd density levels observed"
        
        if crowd_metrics:
            # Calculate risk score based on multiple factors (0-100 scale)
            critical_area_pct = crowd_metrics.get('critical_density_area_pct', 0)
            num_hotspots = crowd_metrics.get('num_hotspots', 0)
            crowd_pattern = crowd_metrics.get('crowd_pattern', 'uniform')
            density_cv = crowd_metrics.get('density_coefficient_variation', 0)
            peak_density = crowd_metrics.get('peak_density', 0)
            hotspot_area_pct = crowd_metrics.get('hotspot_area_pct', 0)
            
            # Risk calculation algorithm
            risk_factors = 0
            
            # Critical density area contribution (0-40 points)
            risk_factors += min(critical_area_pct * 4, 40)
            
            # Number of hotspots contribution (0-20 points)
            risk_factors += min(num_hotspots * 5, 20)
            
            # Clustering pattern contribution (0-20 points)
            pattern_scores = {
                'uniform': 0, 
                'moderate_clustering': 5, 
                'high_clustering': 15, 
                'extreme_clustering': 20
            }
            risk_factors += pattern_scores.get(crowd_pattern, 0)
            
            # Density variation contribution (0-15 points)
            risk_factors += min(density_cv * 7.5, 15)
            
            # Hotspot area contribution (0-5 points)
            risk_factors += min(hotspot_area_pct * 0.5, 5)
            
            stampede_risk_score = min(risk_factors, 100)
            
            # Determine alert level and thresholds
            if stampede_risk_score >= 80:
                alert_level = "Critical"
                alert_threshold_crossed = True
                risk_profile = "CRITICAL_DENSITY_EVENT"
                title = "Critical Crowd Density - Immediate Action Required"
                description = f"Extremely high density detected with {num_hotspots} critical hotspots. Critical area coverage: {critical_area_pct:.1f}%"
            elif stampede_risk_score >= 60:
                alert_level = "High"
                alert_threshold_crossed = True
                risk_profile = "HIGH_DENSITY_EVENT"
                title = "High Crowd Density - Critical Zones Detected"
                description = f"Multiple hotspots identified with critical density thresholds exceeded. Pattern: {crowd_pattern}"
            elif stampede_risk_score >= 40:
                alert_level = "Medium"
                alert_threshold_crossed = True
                risk_profile = "MODERATE_DENSITY_EVENT"
                title = "Moderate Crowd Density - Monitoring Required"
                description = f"Elevated crowd density with {num_hotspots} hotspots detected. Coverage: {crowd_metrics.get('crowd_coverage_pct', 0):.1f}%"
            else:
                alert_level = "Low"
                alert_threshold_crossed = False
                risk_profile = "NORMAL"
                title = "Normal Crowd Conditions"
                description = "Standard crowd density levels observed"

        # Generate highlight regions based on hotspot data
        highlight_regions = []
        if crowd_metrics and crowd_metrics.get('num_hotspots', 0) > 0:
            # Note: This is simplified - in a real implementation you'd need the actual spatial coordinates
            # from the density map analysis to provide accurate bounding boxes
            hotspot_peaks = crowd_metrics.get('hotspot_peak_densities', [])
            for i, peak_density in enumerate(hotspot_peaks[:5]):  # Limit to top 5 hotspots
                # Generate realistic region coordinates (you'd calculate these from actual hotspot locations)
                region = {
                    "regionLabel": f"Hotspot Zone {i+1}",
                    "coordinates": {
                        "x": 50 + (i * 100),  # Simplified positioning
                        "y": 100 + (i * 80),
                        "width": 150 + (peak_density * 10),  # Size based on density
                        "height": 120 + (peak_density * 8)
                    },
                    "riskLevel": "Critical" if peak_density > crowd_metrics.get('critical_threshold', 0) else "High",
                    "density": round(float(peak_density) / crowd_metrics.get('peak_density', 1) if crowd_metrics.get('peak_density', 0) > 0 else 0.5, 2)
                }
                highlight_regions.append(region)

        # Generate observed patterns based on metrics
        observed_patterns = []
        if crowd_metrics:
            if crowd_metrics.get('crowd_pattern') == 'extreme_clustering':
                observed_patterns.extend(["rapid_congregation", "extreme_clustering"])
            elif crowd_metrics.get('crowd_pattern') == 'high_clustering':
                observed_patterns.extend(["congregation_patterns", "clustering_behavior"])
            
            if crowd_metrics.get('critical_density_area_pct', 0) > 5:
                observed_patterns.append("critical_density_zones")
                
            if crowd_metrics.get('num_hotspots', 0) > 3:
                observed_patterns.append("multiple_hotspot_formation")
                
            if crowd_metrics.get('density_coefficient_variation', 0) > 2:
                observed_patterns.append("directional_flow_disruption")

        current_time = datetime.now().isoformat()
        
        # Prepare the comprehensive payload
        payload = {
            "cameraId": camera_id,
            "eventTimestamp": current_time,
            "annotatedImageUrl": annotated_img_payload,
            "rawImageUrl": raw_img_payload,
            
            "stampedeRiskScore": round(float(stampede_risk_score), 0),
            "alertLevel": alert_level,
            "alertThresholdCrossed": alert_threshold_crossed,
            "title": title,
            "description": description,
            "riskProfile": risk_profile,
            
            "highlightRegions": highlight_regions,
            "eventTimestamps": [current_time],
            "cameraViews": ["main_view"],
            "observedPatterns": observed_patterns if observed_patterns else ["normal_flow"],
        }
        
        # Add all crowd metrics if available
        if crowd_metrics:
            # Calculate spread width in meters (simplified calculation)
            # Assuming each pixel represents approximately 0.1 meters (adjust based on camera setup)
            pixels_per_meter = 10  # 10 pixels = 1 meter (adjust for your camera)
            largest_hotspot_size = crowd_metrics.get('largest_hotspot_size', 0)
            spread_width_meters = (largest_hotspot_size ** 0.5) / pixels_per_meter if largest_hotspot_size > 0 else 0
            
            # Normalize peakDensity to 0-1 range
            # Using a reasonable maximum expected density value for normalization
            max_expected_density = 50.0  # Adjust based on your model's typical output range
            raw_peak_density = crowd_metrics.get('peak_density', 0)
            normalized_peak_density = min(raw_peak_density / max_expected_density, 1.0) if max_expected_density > 0 else 0.0
            
            # Round hotspot peak densities to 2 decimal places
            hotspot_peaks_rounded = [round(float(peak), 2) for peak in crowd_metrics.get('hotspot_peak_densities', [])]
            
            payload.update({
                "totalCount": int(crowd_metrics.get('total_count', 0)),
                "peakDensity": round(float(normalized_peak_density), 2),
                "peakHeatmapValue": round(float(normalized_peak_density), 2),  # Same as peakDensity
                "averageDensity": round(float(crowd_metrics.get('average_density', 0)), 2),
                "spreadWidthMeters": round(float(spread_width_meters), 2),
                
                "criticalDensityAreaPercent": round(float(crowd_metrics.get('critical_density_area_pct', 0)), 2),
                "criticalDensityPixels": int(crowd_metrics.get('critical_density_pixels', 0)),
                "criticalThreshold": round(float(crowd_metrics.get('critical_threshold', 0)), 2),
                
                "numberOfHotspots": int(crowd_metrics.get('num_hotspots', 0)),
                "hotspotAreaPercent": round(float(crowd_metrics.get('hotspot_area_pct', 0)), 2),
                "largestHotspotSize": int(crowd_metrics.get('largest_hotspot_size', 0)),
                "averageHotspotSize": round(float(crowd_metrics.get('average_hotspot_size', 0)), 2),
                "hotspotPeakDensities": hotspot_peaks_rounded,
                
                "crowdCoveragePercent": round(float(crowd_metrics.get('crowd_coverage_pct', 0)), 2),
                "crowdPattern": crowd_metrics.get('crowd_pattern', 'uniform'),
                "densityCoefficientOfVariation": round(float(crowd_metrics.get('density_coefficient_variation', 0)), 2),
                
                "noCrowdAreasPercent": round(float(crowd_metrics.get('density_zones_pct', {}).get('no_crowd', 0)), 2),
                "lowDensityAreasPercent": round(float(crowd_metrics.get('density_zones_pct', {}).get('low_density', 0)), 2),
                "mediumDensityAreasPercent": round(float(crowd_metrics.get('density_zones_pct', {}).get('medium_density', 0)), 2),
                "highDensityAreasPercent": round(float(crowd_metrics.get('density_zones_pct', {}).get('high_density', 0)), 2),
                "criticalDensityAreasPercent": round(float(crowd_metrics.get('density_zones_pct', {}).get('critical_density', 0)), 2),
                
                "densityP25": round(float(crowd_metrics.get('density_percentiles', {}).get('p25', 0)), 2),
                "densityP50": round(float(crowd_metrics.get('density_percentiles', {}).get('p50', 0)), 2),
                "densityP75": round(float(crowd_metrics.get('density_percentiles', {}).get('p75', 0)), 2),
                "densityP90": round(float(crowd_metrics.get('density_percentiles', {}).get('p90', 0)), 2),
                "densityP95": round(float(crowd_metrics.get('density_percentiles', {}).get('p95', 0)), 2),
                "densityStandardDeviation": round(float(crowd_metrics.get('density_std', 0)), 2),
                
                "analysisThresholds": {
                    "low": round(float(crowd_metrics.get('thresholds', {}).get('low', 0)), 2),
                    "medium": round(float(crowd_metrics.get('thresholds', {}).get('medium', 0)), 2),
                    "high": round(float(crowd_metrics.get('thresholds', {}).get('high', 0)), 2),
                    "critical": round(float(crowd_metrics.get('thresholds', {}).get('critical', 0)), 2),
                    "adaptive_factor": 1.20,
                    "calibration_timestamp": current_time
                }
            })
        
        # Log the analysis payload structure
        logger.info(f"[{camera_id}] ========== CROWD ANALYSIS PAYLOAD ==========")
        logger.info(f"[{camera_id}] Stampede Risk Score: {round(float(stampede_risk_score), 0)}")
        logger.info(f"[{camera_id}] Alert Level: {alert_level}")
        logger.info(f"[{camera_id}] Alert Threshold Crossed: {alert_threshold_crossed}")
        logger.info(f"[{camera_id}] Risk Profile: {risk_profile}")
        logger.info(f"[{camera_id}] Title: {title}")
        logger.info(f"[{camera_id}] Description: {description}")
        if crowd_metrics:
            spread_width_meters = (crowd_metrics.get('largest_hotspot_size', 0) ** 0.5) / 10 if crowd_metrics.get('largest_hotspot_size', 0) > 0 else 0
            # Calculate normalized peak density for logging
            max_expected_density = 50.0
            raw_peak_density = crowd_metrics.get('peak_density', 0)
            normalized_peak_density = min(raw_peak_density / max_expected_density, 1.0) if max_expected_density > 0 else 0.0
            logger.info(f"[{camera_id}] Peak Heatmap Value (normalized): {round(float(normalized_peak_density), 2)} (raw: {raw_peak_density:.2f})")
            logger.info(f"[{camera_id}] Spread Width Meters: {round(float(spread_width_meters), 2)}m")
        logger.info(f"[{camera_id}] Highlight Regions: {len(highlight_regions)}")
        logger.info(f"[{camera_id}] Observed Patterns: {observed_patterns}")
        logger.info(f"[{camera_id}] Raw Image Data: [BASE64_IMAGE_DATA - {len(raw_img_base64)} chars]")
        logger.info(f"[{camera_id}] Annotated Image Data: [BASE64_IMAGE_DATA - {len(annotated_img_base64)} chars]")
        logger.info(f"[{camera_id}] ==============================================")
        
        # Convert all numpy types to native Python types for JSON serialization
        payload = convert_numpy_types(payload)
        logger.debug(f"[{camera_id}] Payload converted to JSON-serializable format")
        
        # Send POST request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}'
        }
        
        logger.info(f"[{camera_id}] Sending crowd analysis request to {api_endpoint}")
        response = requests.post(api_endpoint,
                               headers=headers,
                               json=payload)
        
        if response.status_code in [200, 201, 202]:
            logger.info(f"[{camera_id}] Successfully sent crowd analysis data. Status: {response.status_code}")
            logger.info(f"[{camera_id}] Risk Score: {round(float(stampede_risk_score), 0)}, Alert Level: {alert_level}")
            if response.text:
                response_text = response.text
                if len(response_text) > 200:
                    logger.info(f"[{camera_id}] Response from crowd analysis API: [LARGE_RESPONSE - {len(response_text)} chars]")
                else:
                    logger.info(f"[{camera_id}] Response from crowd analysis API: {response_text}")
        else:
            logger.error(f"[{camera_id}] Failed to send crowd analysis data. Status code: {response.status_code}")
            if response.text:
                response_text = response.text
                if len(response_text) > 200:
                    logger.error(f"[{camera_id}] Response: [LARGE_ERROR_RESPONSE - {len(response_text)} chars]")
                else:
                    logger.error(f"[{camera_id}] Response: {response_text}")
            
    except Exception as e:
        error_str = str(e)
        if len(error_str) > 300:
            logger.error(f"[{camera_id}] Error sending crowd analysis data: [LARGE_ERROR - {len(error_str)} chars] - {error_str[:300]}...")
        else:
            logger.error(f"[{camera_id}] Error sending crowd analysis data: {error_str}")


def process_frame(cv_img, net, filename, camera_id, api_endpoint, api_token):
    logger.info(f"[{camera_id}] Processing frame: {filename}")
    
    # Convert for model
    img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        logger.debug(f"[{camera_id}] Converted image to RGB mode")
    
    img_tensor = img_transform(img_pil)
    logger.debug(f"[{camera_id}] Image transformed to tensor")

    with torch.no_grad():
        img_tensor = Variable(img_tensor[None, :, :, :]).to(device)
        logger.debug(f"[{camera_id}] Input tensor moved to device: {device}")
        pred_map = net.test_forward(img_tensor)
        logger.debug(f"[{camera_id}] Model forward pass completed")
        
    ''' MAE/MSE'''
    if 'LCM' in model_net:
        pred_value = np.sum(pred_map.cpu().data.numpy()[0, 0, :, :])
    elif 'DM' in model_net:
        pred_value = np.sum(pred_map.cpu().data.numpy()[0, 0, :, :]) / cfg.LOG_PARA 
    
    logger.info(f"[{camera_id}] Predicted count: {pred_value:.2f}")
    
    ''' pred counting map '''
    image = pred_map.cpu().data.numpy()[0, 0, :, :]
    if 'DM' in model_net:
        image = cv2.resize(image, (image.shape[1]*8, image.shape[0]*8))
        logger.debug(f"[{camera_id}] Density map resized for DM model")

    # Calculate comprehensive crowd metrics
    crowd_metrics = calculate_crowd_metrics(image, cv_img.shape[:2], camera_id)
    
    # Create overlay
    original_img = cv_img
    
    # Generate heatmap image data
    norm = plt.Normalize(vmin=np.min(image), vmax=np.max(image))
    cmap = plt.get_cmap('jet')
    heatmap_rgba = cmap(norm(image))
    heatmap_rgb = np.delete(heatmap_rgba, 3, 2)
    heatmap_bgr = cv2.cvtColor((heatmap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    logger.debug(f"[{camera_id}] Heatmap generated")

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap_bgr, (original_img.shape[1], original_img.shape[0]))

    # Blend the images
    alpha = 0.5  # Opacity for the heatmap
    overlay_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_resized, alpha, 0)
    logger.debug(f"[{camera_id}] Overlay image created")
    
    # Send metric data to API with enhanced metrics
    logger.info(f"[{camera_id}] Initiating API call for frame {filename}")
    send_metric_to_api(int(pred_value + 0.5), overlay_img, camera_id, api_endpoint, api_token, crowd_metrics)
    
    # Send crowd analysis data to API
    send_crowd_analysis_to_api(cv_img, overlay_img, camera_id, api_token, crowd_metrics)

    logger.info(f"[{camera_id}] Frame {filename} processing completed")


def process_input(camera_config, net, api_config):
    camera_id = camera_config['id']
    input_url = camera_config['input_url']
    api_endpoint = api_config['endpoint']
    api_token = api_config['token']
    
    logger.info(f"[{camera_id}] Starting processing for input: {input_url}")

    if input_url.startswith('rtsp://'):
        # process stream
        logger.info(f"[{camera_id}] Processing RTSP stream")
        cap = cv2.VideoCapture(input_url)
        if not cap.isOpened():
            logger.error(f"[{camera_id}] Could not open RTSP stream.")
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            logger.warning(f"[{camera_id}] Could not get source FPS. Defaulting to 25 FPS for frame skipping calculation.")
            source_fps = 25  # A reasonable default
        
        target_fps = 0.1
        frame_skip = int(round(source_fps / target_fps))
        logger.info(f"[{camera_id}] Source FPS: {source_fps:.2f}, Target FPS: {target_fps}. Processing 1 frame every {frame_skip} frames.")


        frame_count = 0
        processed_count = 0
        logger.info(f"[{camera_id}] Starting stream processing loop")
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[{camera_id}] Stream ended or failed to read frame.")
                    break
                
                frame_count += 1
                
                if frame_count % frame_skip != 0:
                    logger.debug(f"[{camera_id}] Skipping frame {frame_count}")
                    continue

                logger.info(f"[{camera_id}] Frame {frame_count}: Reading frame from stream for processing")
                
                # Use timestamp for unique filenames
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"stream_frame_{frame_count}_{timestamp}"
                
                process_frame(frame, net, filename, camera_id, api_endpoint, api_token)
                processed_count += 1
                logger.info(f"[{camera_id}] Frame {frame_count} processed successfully. Total processed: {processed_count}")

            except Exception as e:
                logger.error(f"[{camera_id}] Error during frame processing loop: {str(e)}. Skipping to next frame.")

        logger.info(f"[{camera_id}] Stream processing completed. Total frames read: {frame_count}, Total processed: {processed_count}")
        cap.release()
    else:
        # process single image
        logger.info(f"[{camera_id}] Processing single image")
        if not os.path.exists(input_url):
            logger.error(f"[{camera_id}] Image file not found at {input_url}")
            return
        
        img_cv = cv2.imread(input_url)
        if img_cv is None:
            logger.error(f"[{camera_id}] Could not read image {input_url}")
            return
            
        filename = os.path.basename(input_url)
        logger.info(f"[{camera_id}] Processing image: {filename}")
        try:
            process_frame(img_cv, net, filename, camera_id, api_endpoint, api_token)
            logger.info(f"[{camera_id}] Single image processing completed")
        except Exception as e:
            logger.error(f"[{camera_id}] Error processing single image: {str(e)}")


def calculate_crowd_metrics(density_map, image_shape, camera_id):
    """
    Calculate comprehensive crowd metrics from density map
    
    Args:
        density_map: numpy array of crowd density values
        image_shape: tuple of original image dimensions (height, width)
        camera_id: camera identifier for logging
    
    Returns:
        dict: Dictionary containing various crowd metrics
    """
    logger.debug(f"[{camera_id}] Calculating comprehensive crowd metrics")
    
    # Basic statistics
    total_count = np.sum(density_map)
    max_density = np.max(density_map)
    mean_density = np.mean(density_map)
    std_density = np.std(density_map)
    
    # Calculate image area in pixels
    total_pixels = density_map.shape[0] * density_map.shape[1]
    
    # Define density thresholds (adjust based on your model's output range)
    low_threshold = mean_density + 0.5 * std_density
    medium_threshold = mean_density + 1.0 * std_density
    high_threshold = mean_density + 2.0 * std_density
    critical_threshold = mean_density + 3.0 * std_density
    
    # Critical density areas
    critical_mask = density_map > critical_threshold
    critical_density_pixels = np.sum(critical_mask)
    critical_density_percentage = (critical_density_pixels / total_pixels) * 100
    
    # High density areas (hotspots)
    hotspot_mask = density_map > high_threshold
    hotspot_pixels = np.sum(hotspot_mask)
    hotspot_percentage = (hotspot_pixels / total_pixels) * 100
    
    # Find connected components for hotspot analysis
    labeled_hotspots, num_hotspots = ndimage.label(hotspot_mask)
    
    # Calculate hotspot sizes
    hotspot_sizes = []
    hotspot_peak_densities = []
    
    for i in range(1, num_hotspots + 1):
        hotspot_region = labeled_hotspots == i
        hotspot_size = np.sum(hotspot_region)
        hotspot_peak = np.max(density_map[hotspot_region])
        
        hotspot_sizes.append(hotspot_size)
        hotspot_peak_densities.append(hotspot_peak)
    
    # Calculate coverage area (areas with any people)
    people_mask = density_map > (mean_density * 0.1)  # Areas with minimal density
    coverage_pixels = np.sum(people_mask)
    coverage_percentage = (coverage_pixels / total_pixels) * 100
    
    # Density distribution analysis
    non_zero_densities = density_map[density_map > 0]
    density_percentiles = {
        'p25': np.percentile(non_zero_densities, 25) if len(non_zero_densities) > 0 else 0,
        'p50': np.percentile(non_zero_densities, 50) if len(non_zero_densities) > 0 else 0,
        'p75': np.percentile(non_zero_densities, 75) if len(non_zero_densities) > 0 else 0,
        'p90': np.percentile(non_zero_densities, 90) if len(non_zero_densities) > 0 else 0,
        'p95': np.percentile(non_zero_densities, 95) if len(non_zero_densities) > 0 else 0
    }
    
    # Crowd distribution pattern analysis
    # Calculate coefficient of variation to measure clustering vs spreading
    if mean_density > 0:
        density_cv = std_density / mean_density
    else:
        density_cv = 0
    
    # Determine crowd pattern
    if density_cv < 0.5:
        crowd_pattern = "uniform"
    elif density_cv < 1.0:
        crowd_pattern = "moderate_clustering"
    elif density_cv < 2.0:
        crowd_pattern = "high_clustering"
    else:
        crowd_pattern = "extreme_clustering"
    
    # Calculate density zones
    density_zones = {
        'no_crowd': np.sum(density_map <= low_threshold),
        'low_density': np.sum((density_map > low_threshold) & (density_map <= medium_threshold)),
        'medium_density': np.sum((density_map > medium_threshold) & (density_map <= high_threshold)),
        'high_density': np.sum((density_map > high_threshold) & (density_map <= critical_threshold)),
        'critical_density': critical_density_pixels
    }
    
    # Convert to percentages
    density_zones_pct = {k: (v / total_pixels) * 100 for k, v in density_zones.items()}
    
    metrics = {
        # Basic metrics
        'total_count': float(total_count),
        'peak_density': float(max_density),
        'average_density': float(mean_density),
        'density_std': float(std_density),
        
        # Critical density metrics
        'critical_density_area_pct': float(critical_density_percentage),
        'critical_density_pixels': int(critical_density_pixels),
        'critical_threshold': float(critical_threshold),
        
        # Hotspot metrics
        'num_hotspots': int(num_hotspots),
        'hotspot_area_pct': float(hotspot_percentage),
        'hotspot_pixels': int(hotspot_pixels),
        'largest_hotspot_size': int(max(hotspot_sizes)) if hotspot_sizes else 0,
        'average_hotspot_size': float(np.mean(hotspot_sizes)) if hotspot_sizes else 0,
        'hotspot_peak_densities': hotspot_peak_densities,
        
        # Coverage metrics
        'crowd_coverage_pct': float(coverage_percentage),
        'crowd_coverage_pixels': int(coverage_pixels),
        
        # Distribution metrics
        'density_percentiles': density_percentiles,
        'crowd_pattern': crowd_pattern,
        'density_coefficient_variation': float(density_cv),
        
        # Zone analysis
        'density_zones_pct': density_zones_pct,
        'density_zones_pixels': density_zones,
        
        # Thresholds used
        'thresholds': {
            'low': float(low_threshold),
            'medium': float(medium_threshold),
            'high': float(high_threshold),
            'critical': float(critical_threshold)
        }
    }
    
    logger.info(f"[{camera_id}] Metrics calculated - Count: {total_count:.1f}, "
                f"Hotspots: {num_hotspots}, Critical area: {critical_density_percentage:.1f}%, "
                f"Pattern: {crowd_pattern}")
    
    return metrics


def main():
    logger.info("Starting Crowd Counting Demo")

    # Load configuration from yaml file
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded from config.yaml")
    except FileNotFoundError:
        logger.error("config.yaml not found. Please create it.")
        return
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}")
        return

    api_config = config.get('api', {})
    camera_configs = config.get('cameras', [])

    if not camera_configs:
        logger.warning("No cameras found in config.yaml. Exiting.")
        return

    ''' model '''
    logger.info(f"Loading model: {model_net}")
    if 'LCM' in model_net:
        from models.CC_LCM import CrowdCounter
    elif 'DM' in model_net:
        from models.CC_DM import CrowdCounter
    net = CrowdCounter(cfg_GPU_ID, model_net, pretrained=False)
    
    ''' single-gpu / multi-gpu trained model '''
    logger.info(f"Loading model weights from: {model_path}")
    if len(cfg_GPU_ID) == 1:
        net.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Model loaded (single GPU)")
    else:
        load_gpus_to_gpu(net, model_path)
        logger.info("Model loaded (multi-GPU converted to single)")
    
    net.to(device)
    net.eval()
    logger.info("Model loaded and set to evaluation mode")

    threads = []
    for camera_config in camera_configs:
        thread = threading.Thread(target=process_input, args=(camera_config, net, api_config))
        threads.append(thread)
        thread.start()
        logger.info(f"Started processing for camera {camera_config.get('id')}")

    for thread in threads:
        thread.join()

    logger.info("All processing threads completed.")


if __name__ == '__main__':
    logger.info("=== Crowd Counting Demo Started ===")
    main()
    logger.info("=== Crowd Counting Demo Completed ===")