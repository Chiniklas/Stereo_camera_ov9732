DIY stereo camera test environment (dual OV9732 modules on a 3D-printed holder). More build notes and assets coming soon.

## Setup
- Install Miniconda/Anaconda if you don't already have it.
- Create and activate an isolated environment:
  ```bash
  conda create -n stereo_camera python=3.10 -y
  conda activate stereo_camera
  ```
- Install Python requirements:
  ```bash
  pip install -r requirements.txt
  ```

## Single-camera capture
- Config lives in `stereo_camera/cameras/config/ov9732_L.yaml` (or `_R.yaml`).
- Quick run with defaults (prefer the test wrapper):
  ```bash
  python tests/capture_single_stream.py --show
  ```
- Override device/resolution/fps on the fly:
  ```bash
  python tests/capture_single_stream.py \
    --config ov9732_L \
    --device /dev/video4 \
    --width 1280 --height 720 --fps 30 \
    --out left.avi --show --save
  ```
- Behavior: preview is on by default; add `--headless` to disable. Recording happens only when `--save` is set; `--segment N` rotates files; press `q` in the preview to stop.

## Two-camera capture
- Configs: `ov9732_L.yaml` and `ov9732_R.yaml`.
- Quick run with defaults (prefer the test wrapper):
  ```bash
  python tests/capture_two_stream.py --show
  ```
- Override devices/resolution/fps:
  ```bash
  python tests/capture_two_stream.py \
    --config-left ov9732_L --config-right ov9732_R \
    --device-left /dev/video4 --device-right /dev/video2 \
    --width 1280 --height 720 --fps 30 \
    --out-dir recordings --basename stereo --show --save
  ```
- Behavior: preview is on by default; add `--headless` to disable. Recording happens only when `--save` is set; if saving, writes `basename_cam0.avi` and `basename_cam1.avi`; `--segment N` rotates files; `--show` opens a side-by-side preview.

## Notes
- Defaults target MJPG @ 1280x720 @ 30 fps.
- If a camera only supports YUYV/lower FPS, change the `fourcc` to `YUYV` and reduce `--fps`.
