#!/usr/bin/env python3
"""
capture_stability_test.py

Simple, robust test script to capture a stable video stream from a V4L2 USB camera.
Features:
- Uses OpenCV (v4l2 backend) to open `/dev/video*` devices
- Tries to enforce MJPG @ 1280x720 @ 30fps (adjustable via args)
- Detects frame drops and logs them
- Auto-reconnects on camera failures with exponential backoff
- Optional file segmentation (rotate files every N seconds)
- Optional preview window
- Optional flipping if the camera image is inverted (vertical/horizontal/both)

Usage:
  pip3 install opencv-python numpy
  python3 utils/capture_stability_test.py --device /dev/video2 --width 1280 --height 720 --fps 30 --out capture.avi --segment 60 --show --flip vertical

"""

import argparse
import logging
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
import threading

import cv2
import numpy as np


STOP = threading.Event()

# Log files are stored under logs/<timestamp>/<log_name>
def resolve_log_path(log_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / ts / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def sigterm_handler(signum, frame):
    logging.info("Signal %s received, stopping...", signum)
    STOP.set()


signal.signal(signal.SIGINT, sigterm_handler)
signal.signal(signal.SIGTERM, sigterm_handler)


def open_capture(device: str, width: int, height: int, fps: int, fourcc_str: str = "MJPG"):
    logging.info("Opening camera %s (target %dx%d @ %dfps, %s)", device, width, height, fps, fourcc_str)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logging.warning("VideoCapture failed to open %s", device)
        return None

    # Set desired properties
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Read back actual properties
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


def run_capture(args):
    device = args.device
    width = args.width
    height = args.height
    fps = args.fps
    flip = args.flip
    out_prefix = Path(args.out)
    show = args.show
    segment = args.segment

    cap = None
    writer = None
    seq = 0

    consecutive_failures = 0
    max_failures_before_reopen = args.max_fails
    reconnect_wait_base = args.reconnect_wait

    last_frame_time = None
    total_frames = 0
    dropped_frames = 0

    # Stats logging window
    window_times = []
    window_size = 100
    last_stats = time.monotonic()

    def open_writer_for_seq(seq_idx):
        if segment and segment > 0:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_prefix.with_name(f"{out_prefix.stem}_{suffix}.avi")
        else:
            path = out_prefix
        return path, make_writer(path, width, height, "MJPG", fps)

    # initial open
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

    while not STOP.is_set():
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
            # If read fails repeatedly, force reopen
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

        # success
        consecutive_failures = 0
        total_frames += 1

        # optional flipping to correct inverted cameras
        if flip != "none":
            flip_map = {"vertical": 0, "horizontal": 1, "both": -1}
            frame = cv2.flip(frame, flip_map[flip])

        # compute dt & fps
        if last_frame_time is not None:
            dt = now - last_frame_time
            window_times.append(dt)
            if len(window_times) > window_size:
                window_times.pop(0)
        last_frame_time = now

        # write to disk
        if writer:
            writer.write(frame)

        # segmentation
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

        # show preview if requested
        if show:
            cv2.imshow("capture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("'q' pressed, exiting preview")
                STOP.set()
                break

        # periodic stats
        if now - last_stats > args.stats_interval:
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

    # cleanup
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
    parser = argparse.ArgumentParser(description="Camera capture stability test")
    parser.add_argument("--device", default="/dev/video2", help="V4L2 device (e.g., /dev/video2)")
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

    log_path = resolve_log_path(Path(args.log).name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    return run_capture(args)


if __name__ == "__main__":
    sys.exit(main())
