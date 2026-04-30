#!/usr/bin/env python3
"""
24/7 Crowd Analytics Runner
Injects specified cameras into the realtime_crowd module and keeps the session
running indefinitely by auto-restarting when the 8-hour max runtime is reached.
"""

import json
import time
import requests
import sys
import yaml
from pathlib import Path

API = "http://localhost:9010"
AUTH_HEADERS = {}

# Source of truth for the camera list — UI selections (Magicbox picker, manual add,
# Excel upload) all write here via /api/crowd-live/cameras/add.
CONFIG_PATH = Path(__file__).resolve().parent / "py_backend" / "config" / "rtsp_links.yml"


def load_cameras_from_config() -> list:
    """Read crowd_cameras from rtsp_links.yml. Returns [{name, url}, ...]."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[RUNNER] Config not found: {CONFIG_PATH}")
        return []
    except Exception as e:
        print(f"[RUNNER] Failed to read {CONFIG_PATH}: {e}")
        return []

    raw = cfg.get("crowd_cameras") or []
    cams = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        url = (entry.get("url") or "").strip()
        if name and url:
            cams.append({"name": name, "url": url})
    return cams


def login():
    global AUTH_HEADERS
    resp = requests.post(
        f"{API}/api/login",
        json={"username": "admin", "password": "admin123"},
        headers={"x-client-tab": "crowd-247-runner"},
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    AUTH_HEADERS = {
        "Authorization": f"Bearer {token}",
        "x-client-tab": "crowd-247-runner",
    }
    print(f"[RUNNER] Logged in, token={token[:8]}...")


def inject_cameras(cameras: list) -> list:
    """Register each camera via /api/crowd-live/cameras/add (idempotent).
    Returns the list of cam_ids to pass to /start (prefixed with 'cam_').
    """
    cam_ids = []
    added = 0
    skipped = 0
    for cam in cameras:
        name = cam["name"]
        url = cam["url"]
        cam_id = f"cam_{name}"
        cam_ids.append(cam_id)
        try:
            resp = requests.post(
                f"{API}/api/crowd-live/cameras/add",
                json={"name": name, "url": url},
                headers=AUTH_HEADERS,
                timeout=10,
            )
            data = resp.json() if resp.content else {}
            if data.get("status") == "ok":
                added += 1
            elif "already exists" in str(data.get("message", "")).lower():
                skipped += 1
            else:
                print(f"[RUNNER] add_camera {name}: {data}")
        except Exception as e:
            print(f"[RUNNER] add_camera {name} failed: {e}")
    print(f"[RUNNER] Cameras: {added} added, {skipped} existing, {len(cam_ids)} total")
    return cam_ids


def start_crowd(camera_ids):
    resp = requests.post(
        f"{API}/api/crowd-live/start",
        json={"cameras": camera_ids},
        headers=AUTH_HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"[RUNNER] Start response: {data}")
    return data


def stop_crowd():
    try:
        resp = requests.post(
            f"{API}/api/crowd-live/stop",
            headers=AUTH_HEADERS,
        )
        print(f"[RUNNER] Stop response: {resp.json()}")
    except Exception as e:
        print(f"[RUNNER] Stop failed (ok): {e}")


def get_status():
    resp = requests.get(
        f"{API}/api/crowd-live/status",
        headers=AUTH_HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    print("[RUNNER] 24/7 Crowd Analytics Runner starting...")
    print(f"[RUNNER] Camera source: {CONFIG_PATH}")

    login()

    while True:
        try:
            # Reload cameras from rtsp_links.yml — picks up UI selections
            # (Magicbox picker, manual add, Excel upload) on every restart.
            cameras = load_cameras_from_config()
            if not cameras:
                print("[RUNNER] No cameras in crowd_cameras config, sleeping 60s before retry...")
                time.sleep(60)
                continue

            print(f"[RUNNER] Loaded {len(cameras)} camera(s) from config")
            camera_ids = inject_cameras(cameras)

            # Check current status
            status = get_status()
            if status.get("active") or status.get("status") == "running":
                print("[RUNNER] Session already running, stopping first...")
                stop_crowd()
                time.sleep(3)

            # Start the session
            result = start_crowd(camera_ids)

            # Monitor the session — poll every 60s
            while True:
                time.sleep(60)
                try:
                    status = get_status()
                except requests.exceptions.ConnectionError:
                    print("[RUNNER] Backend unreachable, waiting 30s...")
                    time.sleep(30)
                    try:
                        login()
                    except Exception:
                        pass
                    continue
                except requests.exceptions.HTTPError as e:
                    if e.response and e.response.status_code == 401:
                        print("[RUNNER] Token expired, re-logging in...")
                        login()
                        continue

                s = status.get("status", "unknown")
                segs = status.get("segments_completed", 0)
                caps = status.get("total_captures", 0)
                cams = status.get("cameras", 0)
                runtime = status.get("runtime_sec", 0)
                hrs = runtime / 3600

                print(f"[RUNNER] Status={s} | Segments={segs} | Captures={caps} | Cameras={cams} | Runtime={hrs:.1f}h")

                if s in ("completed", "stopped", "error"):
                    print(f"[RUNNER] Session ended ({s}), restarting in 10s...")
                    time.sleep(10)
                    break  # break inner loop to restart

        except KeyboardInterrupt:
            print("\n[RUNNER] Interrupted. Stopping session...")
            stop_crowd()
            sys.exit(0)
        except Exception as e:
            print(f"[RUNNER] Error: {e}, retrying in 30s...")
            time.sleep(30)
            try:
                login()
            except Exception:
                pass


if __name__ == "__main__":
    main()
