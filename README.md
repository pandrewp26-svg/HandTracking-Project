# Hand Tracker

The runnable entry point is `hand_tracker/main.py`. The existing MediaPipe measurements,
CSV session logging, trial tracking, and plots remain in place. Measurements
are not smoothed by Kalman filters.
Depth Anything V2 is an additional relative-depth stream.

## Depth Anything V2 setup

The tracker uses the official [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2)
with the **Small (`vits`) relative-depth model**. At 24.8M parameters, Small is
the lightest released relative-depth checkpoint and is the most practical
starting point for webcam use. It is not guaranteed to run in real time,
especially on CPU. `DEPTH_INPUT_SIZE` in `hand_tracker/main.py` controls the model input
resolution and currently defaults to 320 pixels. Depth inference uses a
background worker with a replaceable latest-frame slot: camera frames continue
to be submitted while inference runs, but pending old frames are replaced
rather than queued.

Install the project's existing dependencies and PyTorch. For PyTorch, use the
official [installation selector](https://pytorch.org/get-started/locally/) to
choose a build compatible with the computer's operating system and GPU. Then,
from PowerShell in the project directory:

```powershell
python -m pip install numpy matplotlib opencv-python mediapipe
python -m pip install torch torchvision
New-Item -ItemType Directory -Force models
git clone https://github.com/DepthAnything/Depth-Anything-V2 models\Depth-Anything-V2
New-Item -ItemType Directory -Force models\weights
Invoke-WebRequest `
  -Uri "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" `
  -OutFile models\weights\depth_anything_v2_vits.pth
```

If PyTorch was installed using the PyTorch selector, do not overwrite that
installation with the generic `pip install torch torchvision` command. The
official inference code and its dependencies are loaded from the cloned
repository; the checkpoint stays separately in `models\weights`.

Expected model files:

```text
models\
  Depth-Anything-V2\       # Official source checkout
  weights\
    depth_anything_v2_vits.pth
```

The Small checkpoint is Apache-2.0 licensed. The official Base and Large model
checkpoints are larger, and are licensed CC-BY-NC-4.0. The official project also
provides metric-depth variants, but they are not used by this tracker.

## Run and controls

From the project directory:

```powershell
python -m hand_tracker.main
# Direct script invocation is supported too:
python .\hand_tracker\main.py
```

- **Q** or **Esc**: quit
- **R**: clear the currently published angle values
- **P**: save the existing measurement plots
- **D**: toggle the side-by-side RGB and relative-depth view
- **G**: turn live finger-bend and palm-normal graphs on or off
- **U**: toggle raw finger bend angle HUD

The MuJoCo window supports mouse orbit and wheel zoom. Camera keys:

- **[ / ]**: rotate left/right
- **, / .**: lower/raise the camera
- **+ / -**: zoom in/out
- **0**: reset camera

The MuJoCo hand uses a fixed 3D point-skeleton template, so detected landmark
positions do not move the model. Raw finger bend angles curl the fingers, while
the raw palm normal controls tilt and the wrist-to-middle-MCP direction resolves
roll. The hand rotates around its palm center, not the wrist. The
thumb tip is landmark 4, separate from the index finger's landmark 5. The
template and bend tuning values are in `hand_tracker/mujoco_hand.py`.

MuJoCo opens a separate 3D window when it is installed. It displays 21
landmark spheres and skeleton connections. The model updates only while a hand
is detected; it stays frozen at its last curl when no hand is detected. A
missing raw angle holds that finger's last bend.

Other modules can read the latest index bend directly; the `filtered` entry is
currently a pass-through alias of the raw value:

```python
from hand_tracker.angles import finger_angles

index_bend = finger_angles["index"]["filtered"]
```

## Angle source

Finger bends and palm-normal axis angles are calculated only from MediaPipe
world landmarks. The existing `<finger>_raw_deg` and
`<finger>_filtered_deg` columns both contain the raw 3D value (the latter is a
pass-through for CSV compatibility);
the appended `<finger>_raw_world_deg` columns record the raw 3D values.
Image landmarks remain used for drawing, depth sampling, and palm orientation.

`hand_tracker/hand_3d.py` supplies the 21-point landmark count and MediaPipe
hand topology for depth-to-3D coordinate conversion.

The depth comparison uses the exact RGB image associated with its result and
labels each of its 21 MediaPipe landmarks with that landmark's relative-depth
median. **NA** marks an invalid sample. A sample is the median of finite
values in the neighborhood; a CSV validity column is `1` only when at least
one finite depth value was available. The camera overlay reports camera FPS,
MediaPipe processing FPS, completed DAV2 inference FPS, average DAV2 inference
time, selected input size, and depth-result age; results older than 2500 ms are
marked stale. CUDA mixed precision is enabled only on CUDA devices. CPU
inference uses PyTorch inference mode without autocast.

The RGB camera window continues to show the tracker whether or not the depth
view is enabled. Depth inference also continues when its visualization is
hidden, so it remains available for logging. The comparison view reports
inference time, inference FPS, and the source-frame age.

The depth display uses the exact MediaPipe landmarks and depth map associated
with each DAV2 source frame. Per-landmark relative-depth scores are logged
alongside the normalized image coordinates and validity flags. None of these
values are calibrated camera-space XYZ or physical distance. For the loaded
Small model, an inference on the official `demo01.jpg` street example gave
larger medians for near foreground regions (for example, foreground car 3.697
and road 3.513) than distant regions (tower 0.162 and horizon road 0.548);
this is an empirical convention check, not a guarantee for every hand/image.

## Depth interpretation and research notes

Depth Anything V2's selected model estimates **relative depth**, not physical
distance. Its raw output is logged unchanged as a relative score; it is not
meters, and its numeric scale is not guaranteed to be consistent between
frames. The visualization independently stretches each map's 2nd-to-98th
percentile to the display range; that display scaling is not applied to logged
values.

Each MediaPipe landmark is mapped from its normalized image coordinates to
the depth map's pixel coordinates. The raw sample is the median of finite
depth values in a clipped 7x7 pixel neighborhood (radius 3 pixels by default),
rather than a single pixel. The landmark's normalized coordinates are scaled
against the depth-map dimensions; the estimator resizes its output to the
source camera dimensions if necessary.
`sample_depth_neighborhood` in `hand_tracker/measurements.py` implements the
sampling, and `DEPTH_SAMPLE_RADIUS` in `hand_tracker/main.py` sets the radius.

The logged `*_dav2_x_norm`, `*_dav2_y_norm`, and `*_dav2_z_relative` columns
combine MediaPipe normalized image X/Y with the raw relative depth score. These
are approximate mixed-coordinate measurements, **not calibrated camera-space
XYZ**. Converting them to physical 3D requires camera calibration and a metric
depth source. The existing MediaPipe relative-Z values and MediaPipe-based
palm-normal calculation remain available. Reserved `*_stereo_depth_m` columns
are left blank for future calibrated stereo measurements.

The raw and legacy filtered Depth Anything CSV columns both contain the raw
relative-depth sample; no Kalman smoothing is applied. Results are associated
with the current detected
MediaPipe landmarks captured with the exact RGB frame sent to inference.
Source frame/age are logged because asynchronous inference may lag the current
camera row; the separate `*_dav2_source_mp_*` columns identify the landmark
coordinates used to sample and construct each result.
The existing CSV column order is retained; source-camera frame, depth-inference
frame, result age, selected resolution, performance statistics, and per-
landmark validity columns are appended.

## Local CPU benchmark

On the development machine (Python 3.13, PyTorch 2.15 CPU, CUDA unavailable),
Depth Anything V2 Small was measured on a synthetic 640x480 BGR frame. After
one warm-up inference, the mean of three runs was:

| Model input size | Mean inference time |
|---:|---:|
| 518 | 2386.1 ms |
| 384 | 1097.3 ms |
| 320 | 674.3 ms |

Each tested inference returned a depth map at the original 640x480 size.
Four sequential asynchronous-worker inferences at 320 averaged 664.5 ms, or
1.562 completed results/sec. MediaPipe processing on 30 synthetic 640x480
frames (after one warm-up frame) measured 46.88 FPS, averaging 21.33 ms/frame.
No live camera benchmark was run. The 320 setting reduces model detail
relative to 518; no ground-truth depth-quality evaluation was performed.
Synthetic timings do not predict the actual live webcam rate. The overlay and
appended CSV performance fields measure live capture, MediaPipe, inference
completion rate, average inference duration, and result age on the system
running the program. Compare recordings at the different sizes before
selecting a research setting.

The model is not hand-specific and may be influenced by background, lighting,
occlusion, motion, and finger boundaries. Relative depth and MediaPipe relative
Z should be treated as separate signals for comparison, not presumed to be
more accurate. The official
[metric-depth models](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
were fine-tuned separately for indoor Hypersim or outdoor Virtual KITTI 2
scenes (with model-specific depth ranges). Their reported meter units do not
guarantee accuracy for an arbitrary webcam/hand setup; using them would require
an explicit model/configuration change and validation.
