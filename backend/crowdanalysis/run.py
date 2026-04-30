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


def send_metric_to_api(count, overlay_img, camera_id, api_endpoint, api_token, event_timestamp=None):
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
        
        # Prepare the payload
        payload = {
            "cameraId": camera_id,
            "metricType": "crowd_count",
            "value_numeric": float(count),
            "unit": "people",
            "timestamp": datetime.now().isoformat(),
            "imageData": img_base64,
            "metadata": {
                "density_level": density_level,
                "flow": "static"
            }
        }
        
        # Add eventTimestamp if provided (for video processing)
        if event_timestamp:
            payload["eventTimestamp"] = event_timestamp
            logger.info(f"[{camera_id}] Using eventTimestamp: {event_timestamp}")
        
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
            if response.status_code == 202:
                logger.info(f"[{camera_id}] Response (202 Accepted): {response.text}")
        else:
            logger.error(f"[{camera_id}] Failed to send metric data. Status code: {response.status_code}")
            logger.error(f"[{camera_id}] Response: {response.text}")
            
    except Exception as e:
        logger.error(f"[{camera_id}] Error sending metric data: {str(e)}")


def send_crowd_analysis_to_api(raw_img, annotated_img, camera_id, api_token, event_timestamp=None):
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

        # Prepare the payload
        payload = {
            "cameraId": camera_id,
            "rawImage": raw_img_payload,
            "annotatedImage": annotated_img_payload
        }
        
        # Add eventTimestamp if provided (for video processing)
        if event_timestamp:
            payload["eventTimestamp"] = event_timestamp
            logger.info(f"[{camera_id}] Using eventTimestamp for crowd analysis: {event_timestamp}")
        
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


def process_frame(cv_img, net, filename, camera_id, api_endpoint, api_token, event_timestamp=None):
    logger.info(f"[{camera_id}] Processing frame: {filename}")
    
    # Preserve original full HD frame for AI analysis (no modifications)
    original_full_hd_frame = cv_img.copy()
    logger.debug(f"[{camera_id}] Original full HD frame preserved: {original_full_hd_frame.shape}")
    
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
    
    # Send metric data to API
    logger.info(f"[{camera_id}] Initiating API call for frame {filename}")
    send_metric_to_api(int(pred_value + 0.5), overlay_img, camera_id, api_endpoint, api_token, event_timestamp)
    
    # Send crowd analysis data to API - using original full HD frame
    send_crowd_analysis_to_api(original_full_hd_frame, overlay_img, camera_id, api_token, event_timestamp)

    logger.info(f"[{camera_id}] Frame {filename} processing completed")


def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from filename in format: camX-YYYY-MM-DD_HH-MM-SS.extension
    Returns datetime object or None if parsing fails
    """
    try:
        # Extract basename without extension
        basename = os.path.splitext(os.path.basename(filename))[0]
        
        # Look for timestamp pattern YYYY-MM-DD_HH-MM-SS in the filename
        # Find the position where the year starts (should be 4 digits)
        import re
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
        match = re.search(timestamp_pattern, basename)
        
        if match:
            timestamp_str = match.group(1)
            # Parse the timestamp: YYYY-MM-DD_HH-MM-SS
            year = int(timestamp_str[0:4])
            month = int(timestamp_str[5:7])
            day = int(timestamp_str[8:10])
            hour = int(timestamp_str[11:13])
            minute = int(timestamp_str[14:16])
            second = int(timestamp_str[17:19])
            
            return datetime(year, month, day, hour, minute, second)
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse timestamp from filename {filename}: {e}")
    
    return None


def process_input(camera_config, net, api_config):
    camera_id = camera_config['id']
    input_url = camera_config['input_url']
    api_endpoint = api_config['endpoint']
    api_token = api_config['token']
    
    # Extract video start time from filename for eventTimestamp calculation
    video_start_time = None
    
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
        
        target_fps = 1
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
    elif input_url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
        # process video file
        logger.info(f"[{camera_id}] Processing video file: {input_url}")
        if not os.path.exists(input_url):
            logger.error(f"[{camera_id}] Video file not found at {input_url}")
            return
        
        # Extract video start time from filename
        video_start_time = extract_timestamp_from_filename(input_url)
        if video_start_time is None:
            logger.warning(f"[{camera_id}] Could not extract timestamp from filename. Using current time as fallback.")
            video_start_time = datetime.now()
        else:
            logger.info(f"[{camera_id}] Extracted video start time from filename: {video_start_time.isoformat()}Z")
            
        cap = cv2.VideoCapture(input_url)
        if not cap.isOpened():
            logger.error(f"[{camera_id}] Could not open video file.")
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0:
            logger.warning(f"[{camera_id}] Could not get source FPS. Defaulting to 25 FPS for calculations.")
            source_fps = 25
        
        target_fps = 1
        frame_skip = int(round(source_fps / target_fps))
        logger.info(f"[{camera_id}] Video FPS: {source_fps:.2f}, Total frames: {total_frames}, Target FPS: {target_fps}")
        logger.info(f"[{camera_id}] Processing 1 frame every {frame_skip} frames")
        logger.info(f"[{camera_id}] Video start time for eventTimestamp: {video_start_time.isoformat()}Z")

        frame_count = 0
        processed_count = 0
        logger.info(f"[{camera_id}] Starting video processing loop")
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"[{camera_id}] Video processing completed. End of file reached.")
                    break
                
                frame_count += 1
                
                if frame_count % frame_skip != 0:
                    logger.debug(f"[{camera_id}] Skipping frame {frame_count}")
                    continue

                logger.info(f"[{camera_id}] Frame {frame_count}: Processing video frame")
                
                # Calculate event timestamp based on video start time and frame position
                seconds_elapsed = frame_count / source_fps
                event_timestamp = (video_start_time + pd.Timedelta(seconds=seconds_elapsed)).isoformat() + 'Z'
                
                filename = f"video_frame_{frame_count}_{int(seconds_elapsed)}s"
                
                process_frame(frame, net, filename, camera_id, api_endpoint, api_token, event_timestamp)
                processed_count += 1
                logger.info(f"[{camera_id}] Frame {frame_count} processed successfully. Event time: {event_timestamp}. Total processed: {processed_count}")

            except Exception as e:
                logger.error(f"[{camera_id}] Error during video frame processing: {str(e)}. Skipping to next frame.")

        logger.info(f"[{camera_id}] Video processing completed. Total frames: {frame_count}, Total processed: {processed_count}")
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