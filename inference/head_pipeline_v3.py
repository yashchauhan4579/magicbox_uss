#!/usr/bin/env python3
"""IRIS Head Detection Pipeline v3 — Zero-Config Auto-Discovery.

Discovers cameras automatically from local USSCore API (localhost:8080).
Discovers device identity from WireGuard interface.
No manual config needed — just deploy and run.

Settings (server URL, thresholds, etc.) in /usr/local/uss/head-pipeline.json
"""

import os, sys, time, threading, signal, json, logging, subprocess, re, base64
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError
import numpy as np
import cv2

from rknnlite.api import RKNNLite

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|fflags;discardcorrupt"
    "|err_detect;bitstream"
    "|stimeout;5000000"        # 5s RTSP socket timeout (µs)
    "|analyzeduration;1000000" # 1s analyze duration (µs) — faster stream open
    "|probesize;500000"        # 500KB probe — faster stream open
)
# Backup: also set read timeout via OpenCV's own mechanism
os.environ["OPENCV_FFMPEG_READ_TIMEOUT_MSEC"] = "5000"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("iris")

GRAY_STD_THRESH = 15.0

stop_flag = False
def _sig(*_):
    global stop_flag; stop_flag = True
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


# ── Config ──────────────────────────────────────────────────────────
# Defaults — overridden by /usr/local/uss/head-pipeline.json if present
DEFAULTS = {
    "report_url": "http://10.100.0.37:9010/api/magicbox-crowd/ingest",
    "model_path": "/usr/local/uss/models/head_fp16_320.rknn",
    "input_size": 320,
    "cycle_interval_sec": 5,
    "rediscovery_interval_sec": 60,
    "conf_threshold": 0.3,
    "iou_threshold": 0.45,
    "thermal_throttle_c": 80,
    "usscore_url": "http://localhost:8080",
}

CONFIG_PATH = "/usr/local/uss/head-pipeline.json"


def load_settings():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                user = json.load(f)
            cfg.update(user)
            log.info(f"Config loaded: {CONFIG_PATH}")
        except Exception as e:
            log.warning(f"Config error ({CONFIG_PATH}): {e}, using defaults")
    else:
        log.info(f"No config at {CONFIG_PATH}, using defaults")
    return cfg


# ── Device Identity ─────────────────────────────────────────────────
def get_device_id():
    """Get WireGuard IP from wg0 interface."""
    try:
        out = subprocess.check_output(
            ["ip", "-4", "addr", "show", "wg0"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        m = re.search(r"inet\s+([\d.]+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    # Fallback: try hostname
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return "unknown"


# ── Camera Discovery from USSCore ───────────────────────────────────
def _unwrap_null(val):
    """Unwrap Go NullString/NullInt64: {"String":"x","Valid":true} → "x" """
    if isinstance(val, dict):
        if "Valid" in val:
            if val["Valid"]:
                return val.get("String", val.get("Int64", val.get("Float64")))
            return None
    return val


def discover_cameras(usscore_url):
    """Fetch cameras from local USSCore API, return [{name, url}]."""
    try:
        req = Request(f"{usscore_url}/api/cameras", headers={"Accept": "application/json"})
        resp = urlopen(req, timeout=10)
        raw = json.loads(resp.read().decode())
    except Exception as e:
        log.warning(f"USSCore discovery failed: {e}")
        return []

    cameras = []
    for cam in raw:
        name = cam.get("name", "unknown")
        # The address field has the full RTSP URL
        rtsp = cam.get("address", "")
        if not rtsp:
            # Try to build from fields
            ip = cam.get("ip", "")
            username = _unwrap_null(cam.get("username", ""))
            password = _unwrap_null(cam.get("password", ""))
            brand = _unwrap_null(cam.get("brand", ""))
            channel = _unwrap_null(cam.get("channel", 1))
            if ip and username and password:
                auth = f"{username}:{password}@{ip}"
                if brand and "hikvision" in str(brand).lower():
                    rtsp = f"rtsp://{auth}:554/Streaming/Channels/{channel}01"
                elif brand and ("dahua" in str(brand).lower() or "cp plus" in str(brand).lower()):
                    rtsp = f"rtsp://{auth}:554/cam/realmonitor?channel={channel}&subtype=0"
                else:
                    rtsp = f"rtsp://{auth}:554/stream1"

        if not rtsp:
            log.warning(f"  Skip {name}: no RTSP URL")
            continue

        # Make name unique if duplicates exist (e.g. NVR channels with same name)
        existing_names = {c["name"] for c in cameras}
        unique_name = name
        if unique_name in existing_names:
            # Try channel number from RTSP URL
            ch_match = re.search(r'[Cc]hannels?/(\d+)', rtsp)
            if ch_match:
                unique_name = f"{name} Ch{ch_match.group(1)}"
            else:
                unique_name = f"{name} #{len(cameras)+1}"
            # Still collides? add counter
            while unique_name in existing_names:
                unique_name = f"{name} #{len(cameras)+1}"

        cameras.append({"name": unique_name, "url": rtsp})

    return cameras


# ── System Utilities ────────────────────────────────────────────────
def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


# ── Camera Thread (persistent connection, always-fresh frame) ──────
class CamKeepAlive:
    def __init__(self, name, url, input_size):
        self.name = name
        self.url = url
        self.sz = input_size
        self.lock = threading.Lock()
        self.npu_input = None
        self.lb_params = None
        self.orig_hw = None
        self.frame_ts = 0.0
        self.running = True
        self.reconnects = 0
        self.connected = False

    def start(self):
        threading.Thread(target=self._run, daemon=True, name=f"cam-{self.name}").start()

    def _letterbox(self, bgr):
        h, w = bgr.shape[:2]
        sz = self.sz
        r = sz / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        dx, dy = (sz - nw) // 2, (sz - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized
        return np.expand_dims(canvas, 0), r, dx, dy

    def _run(self):
        while self.running:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)   # 5s open timeout
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)    # 5s read timeout
            if not cap.isOpened():
                self.connected = False
                # Clear stale frame so inference sees FAIL immediately
                with self.lock:
                    self.npu_input = None
                    self.lb_params = None
                    self.orig_hw = None
                time.sleep(3)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.connected = True
            log.info(f"[{self.name}] connected")

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                if float(np.std(frame[::10, ::10])) < GRAY_STD_THRESH:
                    continue
                npu, r, dx, dy = self._letterbox(frame)
                with self.lock:
                    self.npu_input = npu
                    self.lb_params = (r, dx, dy)
                    self.orig_hw = (frame.shape[0], frame.shape[1])
                    self.frame_ts = time.time()

            cap.release()
            self.connected = False
            # Clear stale frame immediately on disconnect
            with self.lock:
                self.npu_input = None
                self.lb_params = None
                self.orig_hw = None
            if self.running:
                self.reconnects += 1
                log.warning(f"[{self.name}] disconnected, reconnecting in 3s (attempt {self.reconnects})")
                time.sleep(3)

    def snapshot(self, max_age_ms=10000):
        with self.lock:
            if self.npu_input is None:
                return None, None, None, -1
            age = (time.time() - self.frame_ts) * 1000
            if age > max_age_ms:
                # Frame too stale — camera likely disconnected
                return None, None, None, age
            return self.npu_input, self.lb_params, self.orig_hw, age

    def stop(self):
        self.running = False


# ── Postprocessing ──────────────────────────────────────────────────
def postprocess(output, r, dx, dy, conf_thresh, iou_thresh):
    """Returns (count, boxes_letterbox) where boxes_letterbox are in 320x320 space for drawing."""
    pred = output[0]
    if pred.ndim == 3: pred = pred[0]
    if pred.shape[0] == 5: pred = pred.T
    conf = pred[:, 4]
    mask = conf > conf_thresh
    if not np.any(mask):
        return 0, []
    cx, cy, w, h = pred[mask, 0], pred[mask, 1], pred[mask, 2], pred[mask, 3]
    conf = conf[mask]
    # Boxes in original-image coords for NMS
    x1 = ((cx - w/2) - dx) / r
    y1 = ((cy - h/2) - dy) / r
    bw, bh = w / r, h / r
    boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
    confs = conf.tolist()
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, iou_thresh)
    if len(idxs) == 0:
        return 0, []
    kept = idxs.flatten()
    # Return boxes in letterbox space (320x320) for drawing on the npu frame
    lb_boxes = []
    for i in kept:
        lx1 = int(cx[i] - w[i]/2)
        ly1 = int(cy[i] - h[i]/2)
        lx2 = int(cx[i] + w[i]/2)
        ly2 = int(cy[i] + h[i]/2)
        lb_boxes.append((lx1, ly1, lx2, ly2))
    return len(kept), lb_boxes


def encode_frame(npu_input, boxes_lb):
    """Draw boxes on 320x320 frame, JPEG encode, return base64 string."""
    frame = npu_input[0].copy()  # (320, 320, 3)
    for (x1, y1, x2, y2) in boxes_lb:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")


# ── Reporter ────────────────────────────────────────────────────────
def send_report(report_url, device_id, cycle_ms, cam_results, model_name,
                total_cycles, t_start):
    if not report_url:
        return True
    payload = {
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle_ms": round(cycle_ms, 1),
        "cameras": cam_results,
        "system": {
            "temp_c": round(read_temp_c(), 1),
            "uptime_s": round(time.time() - t_start),
            "total_cycles": total_cycles,
            "model": model_name,
        },
    }
    try:
        body = json.dumps(payload).encode()
        req = Request(report_url, data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=5).read()
        return True
    except (URLError, OSError) as e:
        log.warning(f"Report failed: {e}")
        return False


# ── Camera Manager (handles rediscovery) ───────────────────────────
class CameraManager:
    """Manages CamKeepAlive threads. Handles dynamic add/remove on rediscovery."""

    def __init__(self, input_size):
        self.input_size = input_size
        self.cams = {}  # name → CamKeepAlive

    def sync(self, discovered):
        """Sync running cameras with discovered list. Returns (added, removed) counts."""
        discovered_map = {c["name"]: c["url"] for c in discovered}
        added = removed = 0

        # Stop cameras no longer present
        for name in list(self.cams.keys()):
            if name not in discovered_map:
                log.info(f"Camera removed: {name}")
                self.cams[name].stop()
                del self.cams[name]
                removed += 1

        # Start new cameras
        for name, url in discovered_map.items():
            if name not in self.cams:
                log.info(f"Camera added: {name} → {url[:60]}...")
                cam = CamKeepAlive(name, url, self.input_size)
                cam.start()
                self.cams[name] = cam
                added += 1
            elif self.cams[name].url != url:
                # URL changed — restart
                log.info(f"Camera URL changed: {name}")
                self.cams[name].stop()
                cam = CamKeepAlive(name, url, self.input_size)
                cam.start()
                self.cams[name] = cam
                added += 1
                removed += 1

        return added, removed

    def snapshot_all(self):
        """Snapshot all cameras. Returns [(name, npu_in, lb_params, orig_hw, age)]."""
        results = []
        for name, cam in self.cams.items():
            npu_in, lb_params, orig_hw, age = cam.snapshot()
            results.append((name, npu_in, lb_params, orig_hw, age))
        return results

    def stop_all(self):
        for cam in self.cams.values():
            cam.stop()

    @property
    def count(self):
        return len(self.cams)

    @property
    def connected_count(self):
        return sum(1 for c in self.cams.values() if c.connected)


# ── Main ────────────────────────────────────────────────────────────
def main():
    global stop_flag
    cfg = load_settings()

    report_url = cfg["report_url"]
    model_path = cfg["model_path"]
    input_size = cfg["input_size"]
    cycle_sec = cfg["cycle_interval_sec"]
    rediscovery_sec = cfg["rediscovery_interval_sec"]
    conf_thresh = cfg["conf_threshold"]
    iou_thresh = cfg["iou_threshold"]
    thermal_limit = cfg["thermal_throttle_c"]
    usscore_url = cfg["usscore_url"]

    # ── Get device identity ──
    device_id = get_device_id()
    log.info(f"IRIS Pipeline v3 — Auto-Discovery")
    log.info(f"Device: {device_id}")
    log.info(f"Report URL: {report_url}")

    # ── Check model file ──
    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path}")
        sys.exit(1)
    model_name = os.path.basename(model_path)

    # ── Load RKNN ──
    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        log.error(f"load_rknn failed: {model_path}")
        sys.exit(1)
    if rknn.init_runtime() != 0:
        log.error("init_runtime failed")
        sys.exit(1)
    log.info(f"RKNN ready: {model_name}")

    # ── Wait for USSCore with backoff ──
    log.info("Waiting for USSCore...")
    backoff = 3
    while not stop_flag:
        discovered = discover_cameras(usscore_url)
        if discovered:
            log.info(f"Discovered {len(discovered)} cameras from USSCore")
            for c in discovered:
                log.info(f"  {c['name']}: {c['url'][:60]}...")
            break
        log.warning(f"No cameras found, retrying in {backoff}s...")
        for _ in range(backoff):
            if stop_flag: break
            time.sleep(1)
        backoff = min(backoff * 2, 60)

    if stop_flag:
        rknn.release()
        return

    # ── Start camera threads ──
    cam_mgr = CameraManager(input_size)
    cam_mgr.sync(discovered)

    log.info(f"Waiting 5s for camera connections...")
    time.sleep(5)
    for name, cam in cam_mgr.cams.items():
        status = "OK" if cam.connected else "NO FRAMES"
        log.info(f"  {name}: {status} (reconn={cam.reconnects})")

    t_start = time.time()
    total_cycles = 0
    report_ok = 0
    report_fail = 0
    base_cycle = cycle_sec
    last_rediscovery = time.time()

    log.info(f"Pipeline running — {cam_mgr.count} cameras, cycle={cycle_sec}s")

    while not stop_flag:
        cycle_start = time.time()
        total_cycles += 1

        # ── Rediscovery check ──
        if time.time() - last_rediscovery > rediscovery_sec:
            new_cams = discover_cameras(usscore_url)
            if new_cams:
                added, removed = cam_mgr.sync(new_cams)
                if added or removed:
                    log.info(f"Rediscovery: +{added} -{removed} cameras (total={cam_mgr.count})")
                    if added:
                        time.sleep(3)  # Give new cameras time to connect
            last_rediscovery = time.time()

        # ── Phase 1: Snapshot all cameras ──
        snapshots = cam_mgr.snapshot_all()
        grab_ms = (time.time() - cycle_start) * 1000

        # ── Phase 2: Sequential NPU inference ──
        infer_start = time.time()
        cam_results = []
        for (name, npu_in, lb_params, orig_hw, age) in snapshots:
            if npu_in is None:
                cam_results.append({
                    "name": name, "head_count": 0, "inference_ms": 0,
                    "grab_ok": False, "frame_age_ms": -1,
                })
                continue
            r, dx, dy = lb_params
            t0 = time.time()
            outputs = rknn.inference(inputs=[npu_in])
            inf_ms = (time.time() - t0) * 1000
            count, boxes_lb = postprocess(outputs, r, dx, dy, conf_thresh, iou_thresh)
            frame_b64 = encode_frame(npu_in, boxes_lb)
            entry = {
                "name": name, "head_count": count,
                "inference_ms": round(inf_ms, 1), "grab_ok": True,
                "frame_age_ms": round(age, 0),
            }
            if frame_b64:
                entry["frame_b64"] = frame_b64
            cam_results.append(entry)
        infer_ms = (time.time() - infer_start) * 1000
        cycle_ms = (time.time() - cycle_start) * 1000

        # ── Phase 3: Report ──
        ok = send_report(report_url, device_id, cycle_ms, cam_results,
                         model_name, total_cycles, t_start)
        if ok: report_ok += 1
        else: report_fail += 1

        # ── Log ──
        heads = " ".join(
            f"{r['name']}:{r['head_count']}" if r["grab_ok"] else f"{r['name']}:FAIL"
            for r in cam_results
        )
        temp = read_temp_c()
        log.info(f"[cycle {total_cycles}] snap={grab_ms:.0f}ms infer={infer_ms:.0f}ms "
                 f"total={cycle_ms:.0f}ms {temp:.0f}C | {heads}")

        # ── Thermal watchdog ──
        if temp > 90:
            log.warning(f"CRITICAL: {temp:.0f}C, pausing 30s")
            for _ in range(30):
                if stop_flag: break
                time.sleep(1)
            cycle_sec = base_cycle * 2
        elif temp > thermal_limit:
            cycle_sec = base_cycle * 2
        elif cycle_sec > base_cycle:
            cycle_sec = base_cycle

        # ── Sleep until next cycle ──
        elapsed = time.time() - cycle_start
        sleep_time = max(0, cycle_sec - elapsed)
        if sleep_time > 0 and not stop_flag:
            end = time.time() + sleep_time
            while time.time() < end and not stop_flag:
                time.sleep(min(0.5, end - time.time()))

    # ── Shutdown ──
    cam_mgr.stop_all()
    rknn.release()
    elapsed = time.time() - t_start
    log.info(f"Shutdown. {total_cycles} cycles in {elapsed:.0f}s "
             f"({total_cycles/max(elapsed,1)*60:.1f}/min) "
             f"report: {report_ok}ok/{report_fail}fail")


if __name__ == "__main__":
    main()
