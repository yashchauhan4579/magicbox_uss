# MagicBox USS - Head Count Pipeline

Real-time head detection and counting system running on 150+ RK3566 MagicBox edge devices. Each device auto-discovers ONVIF cameras, runs RKNN inference, and reports counts + frames to a central server.

## Architecture

```
[ONVIF Cameras] → [RK3566 Device] → head_pipeline_v3.py → HTTP POST → [Backend Server] → [React Dashboard]
                   (RKNN inference)                                      (FastAPI)
```

## Repo Structure

| Directory | Description |
|-----------|-------------|
| `inference/` | Edge device pipeline script, systemd service, config template |
| `backend/` | Server-side API endpoints (FastAPI) for ingestion, status, frames, hourly stats |
| `frontend/` | React dashboard component for live monitoring and historical analysis |
| `models/` | RKNN model file (Git LFS) - YOLOv8 head detection, FP16, 320px input |
| `deploy/` | Fleet deployment and config management scripts |
| `data/` | Fleet station/device mapping JSON |

## Key Components

### Inference (`inference/head_pipeline_v3.py`)
- Runs on each RK3566 device as a systemd service
- Auto-discovers ONVIF cameras on local network
- Runs RKNN YOLOv8 head detection model (320x320 FP16)
- Reports head counts + JPEG frames to server every 5s
- Config at `/usr/local/uss/head-pipeline.json`

### Backend API (`backend/magicbox_crowd_endpoints.py`)
- `POST /api/magicbox-crowd/ingest` - Receive reports from devices
- `GET /api/magicbox-crowd/status` - Live device status (polled by frontend)
- `GET /api/magicbox-crowd/frames` - Recent detection frames with base64 JPEGs
- `GET /api/magicbox-crowd/hourly` - Hourly aggregated stats from disk metadata
- `GET /api/magicbox-crowd/frames-history` - Historical frames for a date+hour
- `GET /api/magicbox-crowd/fleet` - Station/device location mapping

### Frontend (`frontend/MagicBoxCrowdDashboard.jsx`)
- React component showing live device grid, hourly charts, detection frames
- Station-level filtering with device grouping
- Date picker for historical analysis

## Deployment

Deploy scripts are in `deploy/`. Run from the gateway server (89.116.122.140):

- `mass_deploy.sh` - Full fleet deploy (tarball + config + service install)
- `deploy_remaining.sh` - Retry failed devices
- `update_thresholds.sh` - Push config-only updates fleet-wide (~34s for 140 devices)
- `fix_all_configs.sh` - Push config to specific devices
- `rebuild_deploy.sh` - Rebuild the deploy tarball

Device list: `deploy/magicbox_devices.txt`

## Current Config

```json
{
  "conf_threshold": 0.3,
  "iou_threshold": 0.45,
  "input_size": 320,
  "cycle_interval_sec": 5
}
```

## Model

`models/best_head_fp16_320.rknn` (53MB, tracked via Git LFS)
- YOLOv8 head detection, quantized to FP16 for RK3566 NPU
- Input: 320x320 RGB
- Output: bounding boxes with confidence scores
