"""
MagicBox Crowd - Backend API Endpoints
Extracted from IRIS Command server.py — these are the magicbox-crowd routes
that handle ingestion from edge devices, status, frames, hourly stats, and fleet data.

These endpoints are mounted on the main FastAPI app (Starlette/Uvicorn).
"""

import os
import re
import json
import threading
import collections as _collections
import base64 as _b64
from collections import deque
from datetime import datetime

# ── Globals ──────────────────────────────────────────────────────────
_mb_crowd: dict = {}          # {device_id: {last_report, first_seen, last_seen, total_reports}}
_mb_crowd_lock = threading.Lock()
_mb_crowd_frames = deque(maxlen=100)  # recent frames for /frames API
_mb_fleet_cache: dict = {"data": None, "ts": 0}
_MB_FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "crowd_frames")

_mb_jpeg_queue = _collections.deque()  # FIFO queue of (timestamp, filepath)
_mb_jpeg_queue_loaded = [False]


def _save_frame_to_disk(device_id, camera_name, head_count, timestamp_str, frame_b64):
    """Save ALL frames as JPEG + metadata. FIFO: after 3hrs of data, each new frame
    deletes the oldest one. Metadata JSONL + CSV kept permanently for UI stats."""
    try:
        from datetime import timezone as _tz3, timedelta as _td3
        _IST3 = _tz3(_td3(hours=5, minutes=30))
        _UTC = _tz3(_td3(0))

        ts_clean = timestamp_str.replace("Z", "+00:00") if timestamp_str.endswith("Z") else timestamp_str
        try:
            dt = datetime.fromisoformat(ts_clean)
        except Exception:
            dt = datetime.utcnow()
        # Always ensure timezone-aware (fix naive UTC fallback)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        dt_ist = dt.astimezone(_IST3)
        day_str = dt_ist.strftime("%Y-%m-%d")
        day_dir = os.path.join(_MB_FRAMES_DIR, day_str)
        os.makedirs(day_dir, exist_ok=True)
        safe_dev = device_id.replace(".", "-")
        safe_cam = re.sub(r"[^a-zA-Z0-9_-]", "_", camera_name)[:30]
        ts_file = dt_ist.strftime("%H-%M-%S")
        # Add microsecond hash to prevent filename collisions between cameras
        import hashlib as _hl
        cam_hash = _hl.md5(camera_name.encode()).hexdigest()[:4]
        fname = f"{ts_file}_{safe_dev}_{cam_hash}_{head_count}.jpg"

        # Save JPEG
        fpath = os.path.join(day_dir, fname)
        jpg_bytes = _b64.b64decode(frame_b64)
        with open(fpath, "wb") as f:
            f.write(jpg_bytes)

        # Bootstrap queue on first call (load existing JPEGs from disk)
        if not _mb_jpeg_queue_loaded[0]:
            _mb_jpeg_queue_loaded[0] = True
            import glob as _glob
            cutoff_boot = (dt_ist - _td3(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
            for ddir in sorted(os.listdir(_MB_FRAMES_DIR)):
                dpath = os.path.join(_MB_FRAMES_DIR, ddir)
                if os.path.isdir(dpath) and len(ddir) == 10 and ddir[4] == "-":
                    for jf in sorted(_glob.glob(os.path.join(dpath, "*.jpg"))):
                        bname = os.path.basename(jf)
                        ts_key = ddir + "T" + bname[:8].replace("-", ":")
                        if ts_key < cutoff_boot:
                            # Orphan older than 3hrs — clean it up
                            try:
                                os.remove(jf)
                            except Exception:
                                pass
                        else:
                            _mb_jpeg_queue.append((ts_key, jf))
            print(f"[magicbox-crowd] loaded {len(_mb_jpeg_queue)} JPEGs into queue (cleaned orphans)")

        # Add new frame to queue
        _mb_jpeg_queue.append((day_str + "T" + ts_file.replace("-", ":"), fpath))

        # FIFO: if oldest frame is >3hrs old, delete it (one at a time)
        cutoff_str = (dt_ist - _td3(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
        if _mb_jpeg_queue and _mb_jpeg_queue[0][0] < cutoff_str:
            old_ts, old_path = _mb_jpeg_queue.popleft()
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
            except Exception:
                pass

        # Always append metadata (tiny, kept forever)
        meta = {"file": fname, "device_id": device_id,
                "camera": camera_name, "heads": head_count, "ts": timestamp_str}
        with open(os.path.join(day_dir, "metadata.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")

        # Append to permanent CSV stats (real camera name, not truncated)
        csv_path = os.path.join(_MB_FRAMES_DIR, "crowd_stats.csv")
        csv_exists = os.path.exists(csv_path)
        csv_cam = camera_name.replace(",", ";")  # escape commas for CSV
        with open(csv_path, "a") as f:
            if not csv_exists:
                f.write("date,hour,device_id,camera,heads,timestamp\n")
            f.write(f"{day_str},{dt_ist.hour},{device_id},{csv_cam},{head_count},{timestamp_str}\n")
    except Exception as e:
        print(f"[magicbox-crowd] frame save error: {e}")



@app.post("/api/magicbox-crowd/ingest")
async def api_magicbox_crowd_ingest(request: Request):
    """Receive head count report from an RK3566 edge device. No auth required."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    device_id = body.get("device_id")
    if not device_id:
        raise HTTPException(status_code=400, detail="missing device_id")
    now = body.get("timestamp", datetime.utcnow().isoformat() + "Z")
    with _mb_crowd_lock:
        if device_id not in _mb_crowd:
            _mb_crowd[device_id] = {"first_seen": now, "total_reports": 0}
        _mb_crowd[device_id]["last_report"] = body
        _mb_crowd[device_id]["last_seen"] = now
        _mb_crowd[device_id]["total_reports"] += 1
        # Recent buffer for API + persist ALL to disk
        for cam in body.get("cameras", []):
            fb = cam.get("frame_b64")
            if fb and cam.get("grab_ok"):
                _mb_crowd_frames.append({
                    "device_id": device_id,
                    "camera_name": cam.get("name", "?"),
                    "head_count": cam.get("head_count", 0),
                    "timestamp": now,
                    "frame_b64": fb,
                })
                _save_frame_to_disk(device_id, cam.get("name", "unknown"),
                                    cam.get("head_count", 0), now, fb)
    return {"status": "ok"}

@app.get("/api/magicbox-crowd/status")
def api_magicbox_crowd_status():
    """Return latest data for all reporting devices. Polled by frontend."""
    import time as _time
    now = _time.time()
    with _mb_crowd_lock:
        devices = []
        for did, entry in _mb_crowd.items():
            report = entry.get("last_report", {})
            # Parse last_seen to check if online
            last_seen_str = entry.get("last_seen", "")
            is_online = False
            try:
                from datetime import datetime as _dt, timezone as _tz
                ls = _dt.fromisoformat(last_seen_str.replace("Z", "+00:00"))
                is_online = (now - ls.timestamp()) < 30
            except Exception:
                pass
            cams = [{k: v for k, v in c.items() if k != "frame_b64"} for c in report.get("cameras", [])]
            total_heads = sum(c.get("head_count", 0) for c in cams if c.get("grab_ok"))
            devices.append({
                "device_id": did,
                "is_online": is_online,
                "last_seen": last_seen_str,
                "total_reports": entry.get("total_reports", 0),
                "total_heads": total_heads,
                "cameras": cams,
                "system": report.get("system", {}),
                "cycle_ms": report.get("cycle_ms", 0),
            })
    return {"devices": devices, "total_devices": len(devices)}

@app.get("/api/magicbox-crowd/frames")
def api_magicbox_crowd_frames(request: Request):
    """Return detection frames (newest first). ?limit=N to cap (default 20)."""
    try:
        limit = int(request.query_params.get("limit", 20))
    except (ValueError, TypeError):
        limit = 20
    limit = max(1, min(limit, 100))
    with _mb_crowd_lock:
        frames = list(reversed(_mb_crowd_frames))[:limit]
    return {"frames": frames, "count": len(frames), "total": len(_mb_crowd_frames)}


# ── MagicBox Crowd: Hourly stats + Historical frames ─────────────────
_hourly_cache: dict = {}  # IST-aware  # {"2026-04-29": {"data": [...], "ts": time.time()}}
_hourly_records_cache: dict = {}  # {"2026-04-30": {"records": [...], "ts": time.time(), "size": N}}

@app.get("/api/magicbox-crowd/hourly")
def api_magicbox_crowd_hourly(request: Request):
    """Hourly head count aggregation from disk metadata. ?date=YYYY-MM-DD&device_id=X&camera=Y"""
    import time as _t
    from datetime import timezone as _tz, timedelta as _td
    _IST = _tz(_td(hours=5, minutes=30))
    date = request.query_params.get("date", datetime.now(_IST).strftime("%Y-%m-%d"))
    device_filter = request.query_params.get("device_id", "")
    device_ids_raw = request.query_params.get("device_ids", "")
    device_ids_set = set(d.strip() for d in device_ids_raw.split(",") if d.strip()) if device_ids_raw else set()
    camera_filter = request.query_params.get("camera", "")
    cache_key = f"{date}:{device_filter}:{device_ids_raw}:{camera_filter}"

    # Check cache (60s for today, indefinite for past dates)
    if cache_key in _hourly_cache:
        cached = _hourly_cache[cache_key]
        is_today = date == datetime.now(_IST).strftime("%Y-%m-%d")
        if not is_today or (_t.time() - cached["ts"]) < 60:
            return cached["data"]

    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "crowd_frames", date, "metadata.jsonl")
    if not os.path.exists(meta_path):
        return {"date": date, "hours": [], "total_frames": 0, "peak_hour": -1}

    # Cache parsed records so filtered queries don't re-read 58MB file
    is_today = date == datetime.now(_IST).strftime("%Y-%m-%d")
    file_size = os.path.getsize(meta_path)
    rc = _hourly_records_cache.get(date)
    if rc and (not is_today or (_t.time() - rc["ts"]) < 30) and rc["size"] == file_size:
        all_records = rc["records"]
    else:
        all_records = []
        try:
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts", "")
                    try:
                        _ist_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(_IST)
                        h = _ist_dt.hour
                    except Exception:
                        h = -1
                    all_records.append((rec.get("device_id", ""), rec.get("camera", ""), rec.get("heads", 0), h))
        except Exception as e:
            print(f"[hourly] error reading {meta_path}: {e}")
            return {"date": date, "hours": [], "total_frames": 0, "peak_hour": -1}
        _hourly_records_cache[date] = {"records": all_records, "ts": _t.time(), "size": file_size}

    hours = {h: {"hour": h, "total_heads": 0, "frame_count": 0} for h in range(24)}
    total = 0
    for dev, cam, heads, h in all_records:
        if h < 0:
            continue
        if device_filter and dev != device_filter:
            continue
        if device_ids_set and dev not in device_ids_set:
            continue
        if camera_filter and cam != camera_filter:
            continue
        hours[h]["total_heads"] += heads
        hours[h]["frame_count"] += 1
        total += 1

    hours_list = sorted(hours.values(), key=lambda x: x["hour"])
    peak = max(hours_list, key=lambda x: x["total_heads"])
    result = {"date": date, "hours": hours_list, "total_frames": total, "peak_hour": peak["hour"] if peak["total_heads"] > 0 else -1}
    _hourly_cache[cache_key] = {"data": result, "ts": _t.time()}
    return result


@app.get("/api/magicbox-crowd/frames-history")
def api_magicbox_crowd_frames_history(request: Request):
    """Return historical frames from disk for a specific date+hour. Returns base64 JPEGs."""
    from datetime import timezone as _tz, timedelta as _td
    _IST = _tz(_td(hours=5, minutes=30))
    date = request.query_params.get("date", datetime.now(_IST).strftime("%Y-%m-%d"))
    try:
        hour = int(request.query_params.get("hour", -1))
    except (ValueError, TypeError):
        hour = -1
    device_filter = request.query_params.get("device_id", "")
    device_ids_raw = request.query_params.get("device_ids", "")
    device_ids_set = set(d.strip() for d in device_ids_raw.split(",") if d.strip()) if device_ids_raw else set()
    camera_filter = request.query_params.get("camera", "")
    try:
        limit = int(request.query_params.get("limit", 30))
    except (ValueError, TypeError):
        limit = 30
    limit = max(1, min(limit, 1000))
    try:
        min_heads = int(request.query_params.get("min_heads", 0))
    except (ValueError, TypeError):
        min_heads = 0

    day_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "crowd_frames", date)
    meta_path = os.path.join(day_dir, "metadata.jsonl")
    if not os.path.exists(meta_path):
        return {"frames": [], "count": 0}

    import base64 as _b64
    matching = []
    try:
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rec_dev = rec.get("device_id", "")
                if device_filter and rec_dev != device_filter:
                    continue
                if device_ids_set and rec_dev not in device_ids_set:
                    continue
                if camera_filter and rec.get("camera") != camera_filter:
                    continue
                ts = rec.get("ts", "")
                if hour >= 0:
                    try:
                        _ist_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(_IST)
                        rec_hour = _ist_dt.hour
                    except Exception:
                        continue
                    if rec_hour != hour:
                        continue
                if min_heads > 0 and rec.get("heads", 0) < min_heads:
                    continue
                if min_heads > 0 and rec.get("heads", 0) < min_heads:
                    continue
                matching.append(rec)
    except Exception as e:
        print(f"[frames-history] error: {e}")
        return {"frames": [], "count": 0}

    # Take last N (newest)
    matching = matching[-limit:]
    matching.reverse()

    MAX_JPEG_SIZE = 500 * 1024  # skip files > 500KB (likely corrupted)

    def _read_frame(rec):
        fp = os.path.join(day_dir, rec.get("file", ""))
        b64 = ""
        if rec.get("file") and os.path.exists(fp):
            try:
                sz = os.path.getsize(fp)
                if 0 < sz <= MAX_JPEG_SIZE:
                    with open(fp, "rb") as f:
                        b64 = _b64.b64encode(f.read()).decode("ascii")
            except Exception:
                pass
        return {
            "device_id": rec.get("device_id", ""),
            "camera_name": rec.get("camera", ""),
            "head_count": rec.get("heads", 0),
            "timestamp": rec.get("ts", ""),
            "frame_b64": b64,
        }

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(_read_frame, matching))

    return {"frames": frames, "count": len(frames), "total_matching": len(matching)}


@app.get("/api/magicbox-crowd/fleet")
async def api_magicbox_crowd_fleet():
    """Return static fleet data from JSON file (from magicbox_devices.xlsx)."""
    global _mb_fleet_cache
    if _mb_fleet_cache["data"] is not None:
        return {"stations": _mb_fleet_cache["data"]}
    try:
        fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "magicbox_fleet.json")
        with open(fpath) as f:
            _mb_fleet_cache["data"] = json.load(f)
        return {"stations": _mb_fleet_cache["data"]}
    except Exception as e:
        print(f"[fleet] error loading fleet JSON: {e}")
        return {"stations": []}
