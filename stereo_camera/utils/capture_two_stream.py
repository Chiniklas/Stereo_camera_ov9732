"""
Dual-camera capture helper.

Use `capture_two_stream` to record two USB V4L2 cameras side-by-side with
automatic reconnects, stats logging, optional preview, and optional flipping.
Recording is always full resolution; preview can be downscaled.
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
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

# Hardcoded devices (L then R) matching the user's working setup.
DEFAULT_DEVICES: Sequence[str] = (
    "/dev/video4",  # Left
    "/dev/video2",  # Right
)

# Log files are stored under logs/<timestamp>/<log_name>
def resolve_log_path(log_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / ts / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def label_for_index(idx: int) -> str:
    return "L" if idx == 0 else "R" if idx == 1 else f"Cam{idx}"


def open_capture(device: str, width: int, height: int, fps: int, fourcc_str: str = "MJPG"):
    logging.info("[%s] Opening (target %dx%d @ %dfps, %s)", device, width, height, fps, fourcc_str)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logging.warning("[%s] VideoCapture failed to open", device)
        return None

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    logging.info("[%s] Opened: %dx%d @ %.2ffps", device, actual_w, actual_h, actual_fps)
    return cap


def make_writer(path: Path, width: int, height: int, fourcc_str: str, fps: int, device: str):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    logging.info("[%s] Opening writer %s (codec=%s)", device, path, fourcc_str)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        logging.error("[%s] VideoWriter failed to open %s", device, path)
        return None
    return writer


def _run_capture_thread(
    device: str,
    label: str,
    args,
    out_prefix: Path,
    preview_frames: Dict[str, np.ndarray] | None,
    preview_lock: threading.Lock | None,
):
    width = args.width
    height = args.height
    fps = args.fps
    flip = args.flip
    segment = args.segment
    fourcc_str = "MJPG"

    cap = None
    writer = None
    seq = 0

    consecutive_failures = 0
    max_failures_before_reopen = args.max_fails
    reconnect_wait_base = args.reconnect_wait

    last_frame_time = None
    total_frames = 0
    dropped_frames = 0

    window_times: List[float] = []
    window_size = 100
    last_stats = time.monotonic()

    def open_writer_for_seq(seq_idx):
        if segment and segment > 0:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_prefix.with_name(f"{out_prefix.stem}_{suffix}.avi")
        else:
            path = out_prefix
        return path, make_writer(path, width, height, fourcc_str, fps, device=device)

    backoff = reconnect_wait_base
    cap = open_capture(device, width, height, fps, fourcc_str)
    if cap is None:
        logging.warning("[%s] Initial open failed; retry in %ds", device, backoff)
        time.sleep(backoff)
        cap = open_capture(device, width, height, fps, fourcc_str)
        if cap is None:
            logging.error("[%s] Could not open device; exiting thread", device)
            return

    path, writer = open_writer_for_seq(seq)
    if writer is None:
        logging.error("[%s] Failed to open writer; exiting thread", device)
        return

    while not args.stop_event.is_set():
        if cap is None or not cap.isOpened():
            consecutive_failures += 1
            logging.warning("[%s] Camera closed - attempt reopen (fail #%d)", device, consecutive_failures)
            if consecutive_failures >= max_failures_before_reopen:
                logging.info("[%s] Reopening after %ds backoff", device, backoff)
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                time.sleep(backoff)
                cap = open_capture(device, width, height, fps, fourcc_str)
                backoff = min(backoff * 2, 30)
                if cap is not None and cap.isOpened():
                    logging.info("[%s] Reopened camera", device)
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
            logging.warning("[%s] Frame read failed (consecutive=%d)", device, consecutive_failures)
            if consecutive_failures >= max_failures_before_reopen:
                logging.warning("[%s] Too many frame failures, reopening", device)
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

        if segment and (now - (args.segment_start[label])) >= segment:
            logging.info("[%s] Rotating segment after %.1fs", device, now - args.segment_start[label])
            if writer:
                writer.release()
            seq += 1
            path, writer = open_writer_for_seq(seq)
            if writer is None:
                logging.error("[%s] Failed to open new segment writer; stopping", device)
                break
            args.segment_start[label] = now

        if preview_frames is not None and preview_lock is not None:
            preview_frame = frame.copy()
            cv2.putText(
                preview_frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            with preview_lock:
                preview_frames[label] = preview_frame

        if now - last_stats > args.stats_interval:
            avg_dt = sum(window_times) / len(window_times) if window_times else 0
            avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0
            uptime = now - args.start_time[label]
            logging.info(
                "[%s] uptime=%.1fs total_frames=%d dropped=%d avg_fps=%.2f last_dt=%.3fs",
                device,
                uptime,
                total_frames,
                dropped_frames,
                avg_fps,
                avg_dt,
            )
            last_stats = now

    logging.info("[%s] Stopping: total_frames=%d dropped=%d", device, total_frames, dropped_frames)
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
    if preview_frames is not None and preview_lock is not None:
        with preview_lock:
            preview_frames.pop(label, None)


def capture_two_stream(
    devices: Iterable[str] = DEFAULT_DEVICES,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    out_dir: Path | str = ".",
    basename: str = "capture",
    segment: int = 0,
    show: bool = False,
    log: str = "capture_two.log",
    max_fails: int = 5,
    reconnect_wait: int = 1,
    stats_interval: int = 10,
    flip: str = "vertical",
    preview_scale: float = 0.5,
):
    """
    Capture two cameras in parallel. Returns exit code (0 on success).
    """
    devices = list(devices)
    if len(devices) != 2:
        raise ValueError("Exactly two devices are required")

    log_path = resolve_log_path(Path(log).name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    preview_frames: Dict[str, np.ndarray] | None = {} if show else None
    preview_lock = threading.Lock() if show else None

    stop_event = threading.Event()
    segment_start = {label_for_index(i): time.monotonic() for i in range(len(devices))}
    start_time = {label_for_index(i): time.monotonic() for i in range(len(devices))}

    args = argparse.Namespace(
        width=width,
        height=height,
        fps=fps,
        segment=segment,
        show=show,
        log=log,
        max_fails=max_fails,
        reconnect_wait=reconnect_wait,
        stats_interval=stats_interval,
        flip=flip,
        stop_event=stop_event,
        preview_frames=preview_frames,
        preview_lock=preview_lock,
        segment_start=segment_start,
        start_time=start_time,
    )

    threads = []
    for idx, device in enumerate(devices):
        label = label_for_index(idx)
        out_prefix = out_path / f"{basename}_cam{idx}.avi"
        t = threading.Thread(
            target=_run_capture_thread,
            args=(device, label, args, out_prefix, preview_frames, preview_lock),
            daemon=True,
        )
        threads.append(t)
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            if show and preview_frames is not None and preview_lock is not None:
                with preview_lock:
                    frames: List[np.ndarray] = [preview_frames.get(label_for_index(i)) for i in range(len(devices))]
                if all(f is not None for f in frames):
                    try:
                        resized = [
                            cv2.resize(f, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
                            for f in frames
                        ]
                        composite = cv2.hconcat(resized)
                        cv2.imshow("preview L | R", composite)
                    except Exception as exc:
                        logging.warning("Preview compose failed: %s", exc)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
            for t in threads:
                t.join(timeout=0.05)
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        if show:
            cv2.destroyAllWindows()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Capture two cameras in parallel")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", default=".", help="Directory to place output files")
    parser.add_argument("--basename", default="capture", help="Base filename; index appended per device")
    parser.add_argument("--segment", type=int, default=0, help="If >0, rotate output file every N seconds")
    parser.add_argument("--show", action="store_true", help="Show preview window (side-by-side L|R)")
    parser.add_argument("--log", default="capture_two.log", help="Log file path")
    parser.add_argument("--max-fails", type=int, default=5, help="Max consecutive frame failures before reconnect")
    parser.add_argument("--reconnect-wait", type=int, default=1, help="Initial reconnect wait (seconds) for exponential backoff")
    parser.add_argument("--stats-interval", type=int, default=10, help="Seconds between status logs")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="vertical",
        help="Flip the frame if your camera is inverted (vertical=upside down, horizontal=mirror, both=180° rotate)",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Scale factor for preview window (recording stays full resolution)",
    )
    args = parser.parse_args()

    return capture_two_stream(
        devices=DEFAULT_DEVICES,
        width=args.width,
        height=args.height,
        fps=args.fps,
        out_dir=args.out_dir,
        basename=args.basename,
        segment=args.segment,
        show=args.show,
        log=args.log,
        max_fails=args.max_fails,
        reconnect_wait=args.reconnect_wait,
        stats_interval=args.stats_interval,
        flip=args.flip,
        preview_scale=args.preview_scale,
    )


if __name__ == "__main__":
    sys.exit(main())
