"""
Live stereo preview with rectified RGB and depth (disparity) view.

Loads a saved calibration (.npz), opens both cameras, rectifies frames,
computes disparity with StereoSGBM, and shows a stacked window:
top = rectified RGB (L|R), bottom = disparity heatmap.

Usage:
    python -m stereo_camera.utils.capture_stereo_stream \
        --calib tests/calibration/jetson_stereo.npz \
        --left-config ov9732_L --right-config ov9732_R \
        --flip vertical --preview-scale 0.7

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from stereo_camera.cameras.ov9732_camera import Ov9732Camera
from stereo_camera.utils.capture_two_stream import DualCameraReader


def build_sgbm(
    *,
    min_disp: int,
    num_disp: int,
    block_size: int,
    uniqueness: int,
    speckle_window: int,
    speckle_range: int,
) -> cv2.StereoSGBM:
    # SGBM requires num_disp divisible by 16, block_size odd.
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckle_window,
        speckleRange=speckle_range,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo live view: rectified RGB + disparity.")
    p.add_argument("--calib", default="tests/calibration/jetson_stereo.npz", help="Stereo calibration npz path.")
    p.add_argument("--left-config", default="ov9732_L", help="Left camera config name/path.")
    p.add_argument("--right-config", default="ov9732_R", help="Right camera config name/path.")
    p.add_argument("--flip", choices=["none", "vertical", "horizontal", "both"], default="vertical", help="Flip applied to both streams.")
    p.add_argument("--max-fails", type=int, default=5, help="Consecutive read failures before reconnect.")
    p.add_argument("--reconnect-wait", type=float, default=1.0, help="Initial reconnect backoff (seconds).")
    p.add_argument("--preview-scale", type=float, default=0.7, help="Scale factor for the displayed composite.")
    # SGBM tuning
    p.add_argument("--min-disp", type=int, default=0, help="Minimum disparity.")
    p.add_argument("--num-disp", type=int, default=16 * 8, help="Number of disparities (must be divisible by 16).")
    p.add_argument("--block-size", type=int, default=5, help="SGBM block size (odd, 3-11 typical).")
    p.add_argument("--uniqueness", type=int, default=12, help="Uniqueness ratio (higher = stricter).")
    p.add_argument("--speckle-window", type=int, default=80, help="Speckle window size for post-filtering.")
    p.add_argument("--speckle-range", type=int, default=2, help="Speckle range for post-filtering.")
    p.add_argument("--median", action="store_true", help="Apply 3x3 median blur to disparity for visualization.")
    p.add_argument("--clahe", action="store_true", help="Apply CLAHE to grayscale before matching (helps low contrast).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    calib_path = Path(args.calib)
    if not calib_path.exists():
        print(f"Calibration file not found: {calib_path}", file=sys.stderr)
        return 2

    data = np.load(calib_path)
    k1, d1, k2, d2, R, T = (data[x] for x in ("k1", "d1", "k2", "d2", "R", "T"))

    camL = Ov9732Camera.from_config(args.left_config)
    camR = Ov9732Camera.from_config(args.right_config)
    reader = DualCameraReader([camL, camR], flip=args.flip, max_fails=args.max_fails, reconnect_wait=args.reconnect_wait)
    reader.labels = ["L", "R"]
    reader.start()

    try:
        # Prime one frame to get size
        while True:
            pair = reader.read_pair()
            if pair is None:
                cv2.waitKey(1)
                continue
            h, w = pair[0].shape[:2]
            break

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, (w, h), R, T)
        map1x, map1y = cv2.initUndistortRectifyMap(k1, d1, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(k2, d2, R2, P2, (w, h), cv2.CV_32FC1)
        # Ensure num_disp divisible by 16 and block_size odd
        num_disp = (max(1, args.num_disp // 16)) * 16
        block_size = args.block_size if args.block_size % 2 == 1 else args.block_size + 1
        sgbm = build_sgbm(
            min_disp=args.min_disp,
            num_disp=num_disp,
            block_size=block_size,
            uniqueness=args.uniqueness,
            speckle_window=args.speckle_window,
            speckle_range=args.speckle_range,
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

        while True:
            pair = reader.read_pair()
            if pair is None:
                cv2.waitKey(1)
                continue

            l_raw, r_raw = pair
            l = cv2.remap(l_raw, map1x, map1y, cv2.INTER_LINEAR)
            r = cv2.remap(r_raw, map2x, map2y, cv2.INTER_LINEAR)

            gray_l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            if clahe is not None:
                gray_l = clahe.apply(gray_l)
                gray_r = clahe.apply(gray_r)
            disp = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
            if args.median:
                disp = cv2.medianBlur(disp, 3)

            disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

            rgb_pair = cv2.hconcat([l, r])
            if args.preview_scale != 1.0:
                rgb_pair = cv2.resize(rgb_pair, None, fx=args.preview_scale, fy=args.preview_scale, interpolation=cv2.INTER_AREA)
                disp_vis = cv2.resize(disp_vis, (rgb_pair.shape[1], rgb_pair.shape[0]))

            composite = cv2.vconcat([rgb_pair, disp_vis])
            cv2.imshow("rectified RGB (top) + disparity (bottom) [q to quit]", composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        return 0

    finally:
        reader.stop()


if __name__ == "__main__":
    sys.exit(main())
