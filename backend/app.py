"""Entry point for IRIS Command py_backend.

Sets up multiprocessing queues, starts the relay worker, and launches the FastAPI server.
"""

import os
import time
import threading
import queue

from log_utils import setup_process_logging
from server import (
    run_control_server,
    update_metrics,
    update_frame,
    update_raw_frame,
    get_overlay_state,
    add_alert,
)
from shared_inference import SharedInferenceManager

frame_queue = None
raw_frame_queue = None
metrics_queue = None
alert_queue = None
overlay_shared_dict = None
shared_inference = None


def relay_worker(stop_event, f_q, m_q, a_q, rf_q):
    """Relay metrics, frames, raw frames, and alerts from local queues to control server."""
    while not stop_event.is_set():
        try:
            for _ in range(60):
                if f_q.empty():
                    break
                name, data = f_q.get_nowait()
                update_frame(name, data)
        except:
            pass

        # Relay raw BGR frames; server encodes JPEG once for SAM/MJPEG consumers.
        try:
            for _ in range(60):
                if rf_q.empty():
                    break
                name, data = rf_q.get_nowait()
                update_raw_frame(name, data)
        except:
            pass

        try:
            for _ in range(10):
                if m_q.empty():
                    break
                name, data = m_q.get_nowait()
                update_metrics(name, data)
        except:
            pass

        # Process alert queue
        try:
            for _ in range(5):
                if a_q.empty():
                    break
                source, congestion, metrics_data, screenshot = a_q.get_nowait()
                add_alert(source, congestion, metrics_data, screenshot)
        except:
            pass

        time.sleep(0.002)


def start_backend(idx, url, name, overlay_config=None, active_streams=1):
    """Register a new RTSP source with the shared inference manager."""
    overlay_shared_dict[name] = overlay_config if overlay_config else get_overlay_state(name)
    handle = shared_inference.add_rtsp_stream(idx, name, url, overlay_config)
    if handle is None:
        return None, None
    return handle, handle._stop_event


def start_upload_backend(file_path, name, overlay_config=None, is_crowd=False, active_streams=1, realtime=False):
    """Register an uploaded video with the shared inference manager."""
    overlay_shared_dict[name] = overlay_config if overlay_config else get_overlay_state(name)
    handle = shared_inference.add_upload_stream(file_path, name, overlay_config, is_crowd=is_crowd, realtime=realtime)
    if handle is None:
        return None, None
    return handle, handle._stop_event


def main():
    global overlay_shared_dict, frame_queue, raw_frame_queue, metrics_queue, alert_queue, shared_inference
    setup_process_logging("backend")

    overlay_shared_dict = {}
    frame_queue = queue.Queue(maxsize=60)
    raw_frame_queue = queue.Queue(maxsize=30)
    metrics_queue = queue.Queue(maxsize=10)
    alert_queue = queue.Queue(maxsize=10)

    shared_inference = SharedInferenceManager(
        frame_queue,
        metrics_queue,
        alert_queue,
        raw_frame_queue,
        overlay_shared_dict,
    )
    # Warm models once at startup to reduce first-stream processed delay.
    shared_inference.start()

    stop_relay = threading.Event()
    relay_t = threading.Thread(target=relay_worker, args=(stop_relay, frame_queue, metrics_queue, alert_queue, raw_frame_queue), daemon=True)
    relay_t.start()

    try:
        run_control_server(start_backend, start_upload_backend, overlay_shared_dict)
    finally:
        stop_relay.set()
        relay_t.join(timeout=1.0)
        shared_inference.stop()


if __name__ == "__main__":
    main()
