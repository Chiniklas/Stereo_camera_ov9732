"""
Single-camera capture helper with fixed device mapping.

Choose camera "L" or "R"; devices are hardcoded to the working setup:
L -> /dev/video4, R -> /dev/video2.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# Fixed device map (do not change device nodes; only pick L or R).
DEFAULT_LEFT_DEVICE = "/dev/video4"
DEFAULT_RIGHT_DEVICE = "/dev/video2"
DEVICE_MAP = {"L": DEFAULT_LEFT_DEVICE, "R": DEFAULT_RIGHT_DEVICE}

# Log files are stored under logs/<timestamp>/<log_name>
def resolve_log_path(log_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / ts / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def open_capture(device: str, width: int, height: int, fps: int, fourcc_str: str = "MJPG"):
    logging.info("Opening camera %s (target %dx%d @ %dfps, %s)", device, width, height, fps, fourcc_str)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logging.warning("VideoCapture failed to open %s", device)
        return None

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    logging.info("Camera opened: actual resolution=%dx%d actual_fps=%.2f", actual_w, actual_h, actual_fps)
    return cap


def make_writer(path: Path, width: int, height: int, fourcc_str: str = "MJPG", fps: int = 30):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    logging.info("Opening VideoWriter %s (codec=%s) target fps=%d", path, fourcc_str, fps)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        logging.error("VideoWriter failed to open %s", path)
        return None
    return writer


def capture_single_stream(
    camera: Literal["L", "R"] = "L",
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    out: Path | str = "capture.avi",
    segment: int = 0,
    show: bool = False,
    log: str = "capture.log",
    max_fails: int = 5,
    reconnect_wait: int = 1,
    stats_interval: int = 10,
    flip: str = "vertical",
):
    """
    Capture a single camera stream (select L or R). Returns exit code (0 on success).
    Device nodes are fixed; only the L/R selector may be changed.
    """
    if camera not in DEVICE_MAP:
        raise ValueError("camera must be 'L' or 'R'")
    device = DEVICE_MAP[camera]
    out_path = Path(out)

    log_path = resolve_log_path(Path(log).name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    stop_event = threading.Event()

    def sig_handler(signum, frame):
        logging.info("Signal %s received, stopping...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    cap = None
    writer = None
    seq = 0

    consecutive_failures = 0
    max_failures_before_reopen = max_fails
    reconnect_wait_base = reconnect_wait

    last_frame_time = None
    total_frames = 0
    dropped_frames = 0

    window_times = []
    window_size = 100
    last_stats = time.monotonic()

    def open_writer_for_seq(seq_idx):
        if segment and segment > 0:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_path.with_name(f"{out_path.stem}_{suffix}.avi")
        else:
            path = out_path
        return path, make_writer(path, width, height, "MJPG", fps)

    backoff = reconnect_wait_base
    cap = open_capture(device, width, height, fps, "MJPG")
    if cap is None:
        logging.warning("Initial camera open failed; will retry in %ds", backoff)
        time.sleep(backoff)

    path, writer = open_writer_for_seq(seq)
    if writer is None:
        logging.error("Failed to open writer; exiting.")
        return 1

    start_time = time.monotonic()
    segment_start = start_time

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            consecutive_failures += 1
            logging.warning("Camera closed - attempt reopen (fail #%d)", consecutive_failures)
            if consecutive_failures >= max_failures_before_reopen:
                logging.info("Reopening camera after %ds backoff", backoff)
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                time.sleep(backoff)
                cap = open_capture(device, width, height, fps, "MJPG")
                backoff = min(backoff * 2, 30)
                if cap is not None and cap.isOpened():
                    logging.info("Reopened camera successfully")
                    consecutive_failures = 0
                continue
            else:
                time.sleep(0.1)
                continue

        ret, frame = cap.read()
        now = time.monotonic()

        if not ret or frame is None:
            dropped_frames += 1
            consecutive_failures += 1
            logging.warning("Frame read failed (consecutive fails=%d)", consecutive_failures)
            if consecutive_failures >= max_failures_before_reopen:
                logging.warning("Too many consecutive frame failures, will reopen camera")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                continue
            time.sleep(0.01)
            continue

        consecutive_failures = 0
        total_frames += 1

        if flip != "none":
            flip_map = {"vertical": 0, "horizontal": 1, "both": -1}
            frame = cv2.flip(frame, flip_map[flip])

        if last_frame_time is not None:
            dt = now - last_frame_time
            window_times.append(dt)
            if len(window_times) > window_size:
                window_times.pop(0)
        last_frame_time = now

        if writer:
            writer.write(frame)

        if segment and (now - segment_start) >= segment:
            logging.info("Rotating segment file after %.1fs", now - segment_start)
            if writer:
                writer.release()
            seq += 1
            path, writer = open_writer_for_seq(seq)
            if writer is None:
                logging.error("Failed to open new segment writer; stopping")
                break
            segment_start = now

        if show:
            cv2.imshow(f"capture {camera}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("'q' pressed, exiting preview")
                stop_event.set()
                break

        if now - last_stats > stats_interval:
            avg_dt = sum(window_times) / len(window_times) if window_times else 0
            avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0
            uptime = now - start_time
            logging.info(
                "uptime=%.1fs total_frames=%d dropped=%d avg_fps=%.2f last_dt=%.3fs",
                uptime,
                total_frames,
                dropped_frames,
                avg_fps,
                avg_dt,
            )
            last_stats = now

    logging.info("Stopping capture: total_frames=%d dropped=%d", total_frames, dropped_frames)
    try:
        if cap:
            cap.release()
    except Exception:
        pass
    try:
        if writer:
            writer.release()
    except Exception:
        pass
    if show:
        cv2.destroyAllWindows()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Single camera capture (L or R with hardcoded devices)")
    parser.add_argument("--camera", choices=["L", "R"], default="L", help="Select left or right camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default="capture.avi", help="Output file path or prefix for segments")
    parser.add_argument("--segment", type=int, default=0, help="If >0, rotate output file every N seconds")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--log", default="capture.log", help="Log file path")
    parser.add_argument("--max-fails", type=int, default=5, help="Max consecutive frame failures before reconnect")
    parser.add_argument("--reconnect-wait", type=int, default=1, help="Initial reconnect wait (seconds) for exponential backoff")
    parser.add_argument("--stats-interval", type=int, default=10, help="Seconds between status logs")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="vertical",
        help="Flip the frame if your camera is inverted (vertical=upside down, horizontal=mirror, both=180° rotate)",
    )
    args = parser.parse_args()

    return capture_single_stream(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        out=args.out,
        segment=args.segment,
        show=args.show,
        log=args.log,
        max_fails=args.max_fails,
        reconnect_wait=args.reconnect_wait,
        stats_interval=args.stats_interval,
        flip=args.flip,
    )


if __name__ == "__main__":
    sys.exit(main())
