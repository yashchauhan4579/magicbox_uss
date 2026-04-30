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


def create_json_overlay_data(density_map, original_image_shape, downsample_factor=4):
    """
    Create JSON overlay data that can be used to recreate the heatmap overlay on the client side
    
    Args:
        density_map: numpy array of crowd density values
        original_image_shape: tuple of original image dimensions (height, width)
        downsample_factor: factor to reduce the density map size for efficiency
    
    Returns:
        dict: JSON-serializable dictionary containing overlay data
    """
    logger.debug("Creating JSON overlay data for client-side rendering")
    
    # Downsample density map for efficiency (optional)
    if downsample_factor > 1:
        new_height = density_map.shape[0] // downsample_factor
        new_width = density_map.shape[1] // downsample_factor
        density_map_downsampled = cv2.resize(density_map, (new_width, new_height))
    else:
        density_map_downsampled = density_map
    
    # Calculate normalization parameters
    min_val = float(np.min(density_map))
    max_val = float(np.max(density_map))
    
    # Convert density map to a compressed format
    # Option 1: Send normalized values (0-255 range)
    if max_val > min_val:
        normalized_map = ((density_map_downsampled - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized_map = np.zeros_like(density_map_downsampled, dtype=np.uint8)
    
    # Option 2: For even more compression, you could use run-length encoding or other compression
    # For now, we'll use a simple approach
    
    overlay_data = {
        "type": "heatmap_overlay",
        "density_map": {
            "data": normalized_map.tolist(),  # Convert to list for JSON serialization
            "shape": {
                "height": normalized_map.shape[0],
                "width": normalized_map.shape[1]
            },
            "downsample_factor": downsample_factor
        },
        "normalization": {
            "min_value": min_val,
            "max_value": max_val,
            "has_data": max_val > min_val
        },
        "display_settings": {
            "colormap": "jet",  # Client should use 'jet' colormap
            "alpha": 0.5,      # Opacity for blending with original image
            "interpolation": "bilinear"  # Recommended interpolation method
        },
        "target_dimensions": {
            "height": original_image_shape[0],
            "width": original_image_shape[1]
        },
        "coordinates": {
            "x_offset": 0,
            "y_offset": 0,
            "scale_x": 1.0,
            "scale_y": 1.0
        }
    }
    
    logger.debug(f"JSON overlay data created - Map size: {normalized_map.shape}, "
                f"Range: {min_val:.3f}-{max_val:.3f}, Downsample: {downsample_factor}x")
    
    return overlay_data


def create_optimized_json_overlay_data(density_map, original_image_shape, compression_level=2):
    """
    Create highly optimized JSON overlay data with different compression levels
    
    Args:
        density_map: numpy array of crowd density values
        original_image_shape: tuple of original image dimensions (height, width)
        compression_level: 1=light, 2=medium, 3=heavy compression
    
    Returns:
        dict: JSON-serializable dictionary containing compressed overlay data
    """
    logger.debug(f"Creating optimized JSON overlay data (compression level: {compression_level})")
    
    min_val = float(np.min(density_map))
    max_val = float(np.max(density_map))
    
    if max_val <= min_val:
        # No crowd data
        return {
            "type": "heatmap_overlay",
            "has_data": False,
            "target_dimensions": {
                "height": original_image_shape[0],
                "width": original_image_shape[1]
            }
        }
    
    # Apply compression based on level
    if compression_level == 1:
        # Light compression: downsample by 2x
        downsample_factor = 2
        quantization_levels = 64
    elif compression_level == 2:
        # Medium compression: downsample by 4x
        downsample_factor = 4
        quantization_levels = 32
    else:
        # Heavy compression: downsample by 8x
        downsample_factor = 8
        quantization_levels = 16
    
    # Downsample
    new_height = max(1, density_map.shape[0] // downsample_factor)
    new_width = max(1, density_map.shape[1] // downsample_factor)
    density_map_downsampled = cv2.resize(density_map, (new_width, new_height))
    
    # Quantize values to reduce precision
    normalized_map = (density_map_downsampled - min_val) / (max_val - min_val)
    quantized_map = (normalized_map * (quantization_levels - 1)).astype(np.uint8)
    
    # Find non-zero regions only (sparse representation)
    non_zero_mask = quantized_map > 0
    if np.any(non_zero_mask):
        non_zero_coords = np.column_stack(np.where(non_zero_mask))
        non_zero_values = quantized_map[non_zero_mask]
        
        sparse_data = {
            "coordinates": non_zero_coords.tolist(),
            "values": non_zero_values.tolist()
        }
    else:
        sparse_data = {
            "coordinates": [],
            "values": []
        }
    
    overlay_data = {
        "type": "heatmap_overlay",
        "has_data": True,
        "format": "sparse",
        "density_map": {
            "sparse_data": sparse_data,
            "shape": {
                "height": new_height,
                "width": new_width
            },
            "downsample_factor": downsample_factor,
            "quantization_levels": quantization_levels
        },
        "normalization": {
            "min_value": min_val,
            "max_value": max_val
        },
        "display_settings": {
            "colormap": "jet",
            "alpha": 0.5,
            "interpolation": "bilinear"
        },
        "target_dimensions": {
            "height": original_image_shape[0],
            "width": original_image_shape[1]
        },
        "compression_stats": {
            "original_size": density_map.shape[0] * density_map.shape[1],
            "compressed_size": len(sparse_data["values"]),
            "compression_ratio": len(sparse_data["values"]) / (density_map.shape[0] * density_map.shape[1])
        }
    }
    
    logger.debug(f"Optimized overlay data created - Original: {density_map.shape}, "
                f"Compressed: {new_height}x{new_width}, "
                f"Sparse points: {len(sparse_data['values'])}, "
                f"Compression ratio: {overlay_data['compression_stats']['compression_ratio']:.3f}")
    
    return overlay_data


def send_metric_to_api(count, overlay_img, camera_id, api_endpoint, api_token, crowd_metrics=None, density_map=None, original_image_shape=None):
    try:
        logger.info(f"[{camera_id}] Preparing to send metric data for count: {count}")

        # Convert overlay image to base64
        _, buffer = cv2.imencode('.jpg', overlay_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.debug(f"[{camera_id}] Image converted to base64 successfully")
        
        # Create JSON overlay data if density map is provided
        json_overlay_data = None
        if density_map is not None and original_image_shape is not None:
            json_overlay_data = create_optimized_json_overlay_data(
                density_map, 
                original_image_shape, 
                compression_level=2  # Medium compression
            )
            logger.debug(f"[{camera_id}] JSON overlay data created successfully")

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
            "value_numeric": float(count),
            "unit": "people",
            "timestamp": datetime.now().isoformat(),
            "imageData": img_base64,
            "metadata": base_metadata
        }
        
        # Add JSON overlay data if available
        if json_overlay_data is not None:
            payload["overlayData"] = json_overlay_data
            logger.info(f"[{camera_id}] JSON overlay data added to payload - Format: {json_overlay_data.get('format', 'dense')}, "
                       f"Has data: {json_overlay_data.get('has_data', False)}")
            
            # Log compression statistics if available
            if 'compression_stats' in json_overlay_data:
                stats = json_overlay_data['compression_stats']
                logger.info(f"[{camera_id}] Overlay compression - Original: {stats['original_size']} pixels, "
                           f"Compressed: {stats['compressed_size']} values, "
                           f"Ratio: {stats['compression_ratio']:.3f}")
        else:
            logger.debug(f"[{camera_id}] No JSON overlay data provided - using image-based overlay only")
        
        # Log the complete API payload structure (without image data for readability)
        payload_for_logging = payload.copy()
        payload_for_logging["imageData"] = "[BASE64_IMAGE_DATA]"  # Replace for readability
        logger.info(f"[{camera_id}] ========== API PAYLOAD STRUCTURE ==========")
        logger.info(f"[{camera_id}] Camera ID: {payload['cameraId']}")
        logger.info(f"[{camera_id}] Metric Type: {payload['metricType']}")
        logger.info(f"[{camera_id}] Count Value: {payload['value_numeric']}")
        logger.info(f"[{camera_id}] Unit: {payload['unit']}")
        logger.info(f"[{camera_id}] Timestamp: {payload['timestamp']}")
        logger.info(f"[{camera_id}] Metadata Keys: {list(base_metadata.keys())}")
        logger.info(f"[{camera_id}] Full Metadata:")
        for key, value in base_metadata.items():
            if isinstance(value, dict):
                value_copy_serialized = {key: float(value) if isinstance(value, np.float32) else value for key, value in value.items()}
                type_dict = {key: type(value).__name__ for key, value in value_copy_serialized.items()}
                logger.info(f"type_dict: {type_dict}")
                value_copy_serialized = json.dumps(value_copy_serialized, indent=4)
                logger.info(f"[{camera_id}]   {key}: {value_copy_serialized}")
            else:
                logger.info(f"[{camera_id}]   {key}: {value}")
        logger.info(f"[{camera_id}] ================================================")
        
        # Send POST request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}'
        }
        
        logger.info(f"[{camera_id}] Sending API request to {api_endpoint}")

        def convert_npfloat32(obj):
            if isinstance(obj, dict):
                return {key: convert_npfloat32(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_npfloat32(item) for item in obj]
            elif isinstance(obj, np.float32):
                return float(obj)
            else:
                return obj
        #payload = {key: float(value) if isinstance(value, np.float32) else value for key, value in payload.items()}
        payload = convert_npfloat32(payload)

        type_dict = {key: type(value).__name__ for key, value in payload.items()}
        logger.info(f"payload_type_dict:{type_dict}")
        response = requests.post(api_endpoint,
                               headers=headers,
                               json=payload)
        
        if response.status_code in [200, 202]:
            logger.info(f"[{camera_id}] Successfully sent metric data. Count: {count}, Status: {response.status_code}")
            if response.status_code == 202:
                logger.info(f"[{camera_id}] Response (202 Accepted): {response.text}")
        else:
            logger.error(f"[{camera_id}] Failed to send metric data. Status code: {response.status_code}")
            logger.error(f"[{camera_id}] Response: {response.text}")
            
    except Exception as e:
        logger.error(f"[{camera_id}] Error sending metric data: {str(e)}")


def send_crowd_analysis_to_api(raw_img, annotated_img, camera_id, api_token, density_map=None, original_image_shape=None):
    try:
        api_endpoint = 'http://localhost:3000/api/v1/ai/crowd-analysis'
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

        # Create JSON overlay data if density map is provided
        json_overlay_data = None
        if density_map is not None and original_image_shape is not None:
            json_overlay_data = create_optimized_json_overlay_data(
                density_map, 
                original_image_shape, 
                compression_level=2  # Medium compression
            )
            logger.debug(f"[{camera_id}] JSON overlay data created for crowd analysis API")

        # Prepare the payload
        payload = {
            "cameraId": camera_id,
            "rawImage": raw_img_payload,
            "annotatedImage": annotated_img_payload
        }
        
        # Add JSON overlay data if available
        if json_overlay_data is not None:
            payload["overlayData"] = json_overlay_data
            logger.info(f"[{camera_id}] JSON overlay data added to crowd analysis payload - Format: {json_overlay_data.get('format', 'dense')}, "
                       f"Has data: {json_overlay_data.get('has_data', False)}")
            
            # Log compression statistics if available
            if 'compression_stats' in json_overlay_data:
                stats = json_overlay_data['compression_stats']
                logger.info(f"[{camera_id}] Crowd analysis overlay compression - Original: {stats['original_size']} pixels, "
                           f"Compressed: {stats['compressed_size']} values, "
                           f"Ratio: {stats['compression_ratio']:.3f}")
        else:
            logger.debug(f"[{camera_id}] No JSON overlay data provided for crowd analysis - using image-based overlay only")
        
        # Send POST request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}'
        }
        
        logger.info(f"[{camera_id}] Sending API request to {api_endpoint}")
        response = requests.post(api_endpoint,
                               headers=headers,
                               json=payload)
        
        if response.status_code in [200, 201, 202]:
            logger.info(f"[{camera_id}] Successfully sent crowd analysis data. Status: {response.status_code}")
            if response.text:
                logger.info(f"[{camera_id}] Response from crowd analysis API: {response.text}")
        else:
            logger.error(f"[{camera_id}] Failed to send crowd analysis data. Status code: {response.status_code}")
            logger.error(f"[{camera_id}] Response: {response.text}")
            
    except Exception as e:
        logger.error(f"[{camera_id}] Error sending crowd analysis data: {str(e)}")


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
    send_metric_to_api(
        int(pred_value + 0.5), 
        overlay_img, 
        camera_id, 
        api_endpoint, 
        api_token, 
        crowd_metrics,
        density_map=image,  # Pass the density map for JSON overlay creation
        original_image_shape=cv_img.shape[:2]  # Pass original image dimensions
    )
    
    # Send crowd analysis data to API
    send_crowd_analysis_to_api(
        cv_img, 
        overlay_img, 
        camera_id, 
        api_token,
        density_map=image,  # Pass the density map for JSON overlay creation
        original_image_shape=cv_img.shape[:2]  # Pass original image dimensions
    )

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
