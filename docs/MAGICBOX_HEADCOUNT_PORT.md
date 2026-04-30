# Magicbox Head-Count Port — Replication Guide

Port of a YOLOv8-style head-detection model to the Rockchip RK3566 "magicbox"
edge device (1 TOPS NPU). Runs 1 FPS head detection across 3–4 RTSP cameras
on the NPU alone, no cloud calls.

---

## What you get when this is set up

- **1 FPS per camera × 4 cameras** head detection on the RK3566 NPU
- **FP16 model at 320×320** input, ~195 ms/inference single-stream on the NPU
- **Head-count accuracy matches a GPU FP32 baseline** (same model, ultralytics)
  to within a detection or two per frame
- Optional: annotated MP4 output with live overlays, for demos

Measured benchmark on the reference box (board at 60 s mark of a 120 s run):

| cam | res | codec | inf/s | avg_heads | avg_inf_ms |
| --- | --- | --- | --- | --- | --- |
| magicbox_4 | 960×576 | h264 | 1.00 | 3.22 | 237.6 |
| feed1 | 1920×1080 | h264 | 0.98 | 1.12 | 237.9 |
| feed3 | 1920×1080 | h264 | 0.88 | 0.77 | 239.8 |
| feed7 | 1920×864 | h264 | 0.94 | 0.11 | 239.3 |
| **total** | | | **3.80 / 4.0 target** | | |

Thermals: CPU +19 °C over 2 min with SW decode (51 → 70 °C). **Add a heatsink
before sustained operation** — a bare board will throttle around 85 °C.

---

## Hardware & OS requirements

Verified on:
- Rockchip RK3566 (`rockchip,rk3566-evb2-lp4x-v10`), 4× Cortex-A55 @ 1.8 GHz
- 2 GB RAM, kernel 5.10, Debian 11 bullseye
- RKNPU driver v0.9.8, librknnrt 2.3.2

Other RK3566/RK3568 boards with the same driver should work; RK3588 needs the
model re-converted for its 3-core NPU.

**Disk note**: on the reference box, `/` is small (~5.9 GB) and easily fills.
Install into `/userdata/` (which typically has 20+ GB free). This package is
designed to live wholly under one directory — no `/usr/local/` writes.

---

## 1 — Install

On the magicbox:

```bash
# Put the package somewhere with disk space
cd /userdata/linaro    # or wherever you like
unzip magicbox_headcount.zip
cd magicbox_headcount
bash install.sh
```

`install.sh` creates `.venv/`, installs `numpy` + `opencv-python-headless`,
tries to install `rknn-toolkit-lite2` from `wheels/` if you've dropped a
wheel there, and symlinks system GStreamer bindings into the venv so the
optional HW-decode runner works.

### rknn-toolkit-lite2 wheel (if `install.sh` says it's missing)

Download the wheel matching your Python version + arch from Rockchip's
GitHub, drop it into `wheels/`, re-run `install.sh`:

```
https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages
```

For Python 3.9 / aarch64 (the reference box) the file name looks like
`rknn_toolkit_lite2-2.3.2-cp39-cp39-linux_aarch64.whl`.

### GStreamer plugins (only needed if you want HW decode)

```bash
sudo apt install \
  python3-gi \
  gstreamer1.0-rockchip1 \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-good
```

Verify `mppvideodec` is available:
```bash
gst-inspect-1.0 mppvideodec | head -3
```

---

## 2 — The model

Drop `best_head_fp16_320.rknn` into `models/`. Size is ~53 MB.

### If you already have a converted `.rknn`
Copy it from your existing magicbox:
```bash
scp user@old_magicbox:/path/to/best_head_fp16_320.rknn models/
```

### If you're starting from the PyTorch `best_head.pt`
Do this **on a GPU workstation, not the magicbox** — the conversion tooling
needs an x86 Linux box. You need `rknn-toolkit2` (full, not "lite"):

```bash
# On your GPU workstation, in a separate venv:
pip install rknn-toolkit2  # or install from Rockchip GitHub
```

1. Export PyTorch → ONNX (use ultralytics' exporter, opset 12, **static**
   shape 1×3×320×320). A dynamic-shape ONNX will break RKNN.
2. Feed the ONNX into `rknn-toolkit2` targeting `rk3566`, precision FP16,
   no quantization calibration (FP16 doesn't need it).
3. Ship the resulting `.rknn` to the magicbox.

FP16 was chosen because the accuracy matched the GPU FP32 baseline
(~3 heads/frame on a busy street scene) without needing calibration images.
INT8 would give ~2× the NPU throughput if you need to scale past 4 cameras,
but needs 100+ representative frames for calibration.

---

## 3 — Configure your cameras

Every runner has a hardcoded list near the top:

```python
CAMERAS = [
    ("magicbox_4", "rtsp://user:pass@host:554/path1"),
    ("feed1",      "rtsp://user:pass@host:554/path2"),
    ("feed3",      "rtsp://user:pass@host:554/path3"),
    ("feed7",      "rtsp://user:pass@host:554/path4"),
]
```

Edit in place. The defaults point at a public test relay; replace them with
**your** camera URLs before running for real.

RTSP passwords containing special characters (`*`, `@`, `$`, etc.) must be
URL-encoded (e.g. `*` → `%2A`, `@` → `%40`, `$` → `%24`).

---

## 4 — Run it

### Primary runner — 4 cameras, SW decode (most stable)

```bash
.venv/bin/python scripts/head_runner_multi.py --duration 120
```

Prints a row per inference with per-camera head count, staleness, and
inference time, plus a health line (CPU temp, RAM) every 10 s and a
summary table at the end.

### Single camera (for quick sanity check)

```bash
.venv/bin/python scripts/head_runner.py "rtsp://..." --fps 1 --max-frames 30
```

### Live demo with MP4 output (for showing off)

```bash
.venv/bin/python scripts/head_demo.py --duration 60 --composite-fps 1
```

Writes `/userdata/linaro/head_demo_output.mp4` (1280×760, 1 FPS, ~9 MB for
60 s) with a 2×2 grid of the 4 feeds, green bounding boxes, per-cell head
counts, and a top banner showing inference rate + thermals. Pass `--out
./demo.mp4` to change the output path.

### HW-decode variant (experimental)

```bash
.venv/bin/python scripts/head_runner_multi_hw.py --duration 120
```

Uses GStreamer + `mppvideodec` to offload H.264 decode to the VPU. Gives
~20 % lower inference time under 4-cam load and ~60 % less thermal rise
over 2 min. **Caveat:** on unstable/corrupted RTSP streams, the pipeline
is less forgiving than OpenCV — expect more reconnects until you tune it.
On clean production cameras it's the right choice.

### Useful scripts for debugging

- `scripts/inspect_model.py` — loads the `.rknn`, runs inference on one
  frame of a sample MP4, prints output tensor shapes + single-stream FPS.
  Run once after install to confirm the NPU is wired up.
- `scripts/hw_diag.py` — captures one HW-decoded frame per camera and
  saves as JPG. Used for validating GStreamer pipelines without touching
  the inference path.
- `scripts/gst_hw_test.sh` — benchmarks HW decode throughput standalone
  (no Python, no NPU); useful for confirming `mppvideodec` is working.

---

## 5 — Expected output

Smoke test (`inspect_model.py`):

```
[1/4] Loading .../best_head_fp16_320.rknn
[2/4] init_runtime (NPU core auto)
[3/4] Reading 1 frame from .../vcc_recording_5min.mp4
    frame shape: (1440, 2560, 3)
    model input: (1, 320, 320, 3), dtype=uint8
[4/4] Warmup + timed inference (5 runs)
    output[0]: shape=(1, 5, 2100), dtype=float32, min=-0.312, max=387.250, mean=82.508

Avg inference: 196.9 ms/frame  ->  5.1 FPS single-stream
```

Key numbers: inference ≤ 210 ms, output shape `(1, 5, 2100)`.

4-cam runner (`head_runner_multi.py`):

```
[init] loading model: ...
[init] starting 4 capture threads
[init] 4/4 cameras have frames: [...]
[run] header: elapsed  cam  heads  staleness_ms  inf_ms  reconnects
   5.02  cam1           2      110    198.3  rc=0
   5.27  cam2           3       95    197.1  rc=0
   ...
[health t=10s] temp=[55,50]C  avail_mem=1350MB  npu_load=...
   ...
===== SUMMARY over 120.1s =====
camera         n  avg_heads  avg_inf_ms  avg_stale_ms  reconnects
cam1         120       3.22       237.6           300           0
...
total inferences: 457   achieved 3.80 inferences/sec (target 4.0)
```

---

## 6 — Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `load_rknn failed` | `.rknn` not found or wrong SoC | Check `models/`, confirm SoC matches |
| `init_runtime failed` | NPU driver mismatch | `cat /sys/kernel/debug/rknpu/version` should say ≥ 0.9.x |
| `No module named rknnlite` | Wheel not installed into venv | Drop wheel in `wheels/`, re-run `install.sh` |
| `No module named gi` (HW runner only) | System `gi` not symlinked | Re-run `install.sh`, or `sudo apt install python3-gi` first |
| RTSP opens but gives 0 frames | OpenCV default is UDP transport | All included runners already set `rtsp_transport=tcp`; confirm that env var is set |
| `h264 error while decoding MB...` warnings | Corrupted RTSP stream from relay | Cosmetic — SW decode recovers silently; HW decode may reconnect |
| Inference time creeps over time | Thermal throttle | Check `/sys/class/thermal/thermal_zone*/temp`; add a heatsink |
| "No such device" on `/dev/video-dec0` | Kernel doesn't expose VPU as v4l2-m2m on this image | Known — use the GStreamer `mppvideodec` path (HW runner) instead of FFmpeg's `h264_v4l2m2m` |

---

## 7 — How the inference path works

```
RTSP camera(s)
     ↓
  OpenCV VideoCapture (one thread per camera, keeps only the latest frame)
     ↓                    OR, HW variant:
     ↓               rtspsrc → rtph264depay → h264parse → mppvideodec (VPU)
     ↓                    → videoconvert → appsink
     ↓
  Main loop (round-robin through cameras)
     ↓
  Letterbox 320×320 + BGR→RGB
     ↓
  RKNNLite.inference()  →  output (1, 5, 2100) float32
     ↓
  Postprocess: threshold on ch[4], convert xywh → xyxy, NMS (cv2.dnn.NMSBoxes)
     ↓
  Head count, per-camera rate counters, optional frame overlay + MP4 write
```

All timing is controlled by a per-camera scheduler: each camera has a
`next_tick` and the main loop picks the camera with the earliest tick. This
gives steady 1 FPS per camera regardless of inference jitter.

---

## 8 — What's explicitly NOT included

- **Crowd density (CCN)** — dropped per spec; only head-detection runs here.
- **Gemini** or any cloud analytics — fully local.
- **Any HTTP server** — runners print to stdout and (optionally) write MP4.
  Plumb the head counts into your downstream system however you like.
- **HW encode** — the demo MP4 uses OpenCV's `mp4v` codec (software). For
  prod you'd use `mpph264enc` via a GStreamer pipeline in the VideoWriter.

---

## 9 — File layout

```
magicbox_headcount/
├── README.md                          # this file
├── install.sh                         # set up venv + deps
├── requirements.txt                   # pip deps (rknnlite handled separately)
├── scripts/
│   ├── head_runner.py                 # single camera
│   ├── head_runner_multi.py           # 4 cameras, SW decode — PRIMARY
│   ├── head_runner_multi_hw.py        # 4 cameras, HW decode — experimental
│   ├── head_demo.py                   # 2x2 grid with overlays, saves MP4
│   ├── inspect_model.py               # smoke-test the .rknn on one frame
│   ├── hw_diag.py                     # HW-decode pipeline diagnostic
│   └── gst_hw_test.sh                 # GStreamer HW-decode benchmark
├── models/
│   └── best_head_fp16_320.rknn        # ← drop the model here
└── wheels/
    └── rknn_toolkit_lite2-*.whl       # ← drop the wheel here
```

---

## 10 — Reference performance notes

- **Model input**: 320×320 (small) — chose this over 640 because it's 4× faster
  and the accuracy drop on real drone/street feeds was negligible.
- **Model output**: `(1, 5, 2100)` = single-class YOLOv8-style output.
  5 channels = `[cx, cy, w, h, head_conf]`; 2100 anchors = 40×40 + 20×20 + 10×10
  at strides 8/16/32.
- **Single-stream throughput on RK3566**: 5.1 FPS FP16 (from `inspect_model.py`
  benchmark).
- **Multi-cam ceiling**: ~4 cameras × 1 FPS comfortably. 5+ requires INT8.
- **RTSP relay reality**: public test relays can emit mid-stream h264 corruption
  ("reference picture missing", "cabac decode failed"). Production cameras on
  a LAN don't have this — disregard the decoder warnings unless they coincide
  with actual frame loss.
