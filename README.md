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
```bash
python tests/capture_single_stream.py \
  --camera L \  # or R (devices hardcoded)
  --width 1280 --height 720 --fps 30 \
  --out /tmp/capture.avi \
  --stats-interval 5
```

Common flags:
- `--show` display a preview window (press `q` to stop preview).
- `--segment N` rotate output files every `N` seconds (useful for long runs).
- `--max-fails` consecutive frame failures before reopening the camera.
- `--reconnect-wait` initial backoff (seconds) when reopening the camera.
- `--flip` correct orientation: default is `vertical`. Options: `vertical` (upside-down), `horizontal` (mirror), `both` (180° rotate), or `none`.

## Dual-camera capture
```bash
python tests/capture_two_stream.py \
  --width 1280 --height 720 --fps 30 \
  --out-dir /tmp \
  --basename stereo \
  --stats-interval 5
```
- Devices are fixed (no config): left = `/dev/video4`, right = `/dev/video2`.
- Outputs: `/tmp/stereo_cam0.avi`, `/tmp/stereo_cam1.avi`.
- Default flip is `vertical`; change with `--flip`.
- Add `--show` to open a single preview window with one row/two columns labeled **L** and **R**; press `q` to stop all streams.
- Logs are stored under `logs/<timestamp>/capture_two.log`.

## Import and use as a package
Install in editable mode:
```bash
pip install -e .
```

Example use in Python:
```python
from stereo_camera.utils.capture_two_stream import capture_two_stream, DEFAULT_DEVICES

# run with defaults (L=/dev/video4, R=/dev/video2)
capture_two_stream(devices=DEFAULT_DEVICES, out_dir="/tmp", basename="stereo", show=False)

# or change preview scale only
capture_two_stream(preview_scale=0.4, show=True)

# single-camera (choose L or R; device nodes are hardcoded)
from stereo_camera.utils.capture_single_stream import capture_single_stream

capture_single_stream(camera="L", out="/tmp/left.avi", show=False)
# Logs are written to `logs/<timestamp>/capture.log` by default.
```

Logs live in `logs/<timestamp>/capture*.log`. Scripts report uptime, total frames, dropped frames, and estimated FPS every `--stats-interval` seconds.

## Notes
- Defaults target MJPG @ 1280x720 @ 30 fps.
- If a camera only supports YUYV/lower FPS, change the `fourcc` to `YUYV` and reduce `--fps`.
