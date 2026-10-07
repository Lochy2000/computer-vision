# Drone Detection v3

An experimental, model-agnostic benchmark for long-range drone detection in RGB
and thermal video. The first milestone provides a stable detector interface,
video ingestion, timing, class filtering and JSONL output. Tracking, tiled
inference and sensor fusion will build on the same detection schema.

## Current scope

- Local video files, streams, or camera IDs through OpenCV
- Ultralytics-compatible local `.pt` weights
- Configurable confidence threshold and inference resolution
- Optional model-class filtering
- One JSON record per frame, including inference time and full-frame boxes

The project deliberately does not automatically download weights or datasets.
This prevents accidental use of unverified or non-commercial training material.
See [DATASETS.md](DATASETS.md) for the licence register.

## Install

From this directory:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[ultralytics,dev]"
```

## Run

On Windows, prefer `run.ps1`; it always selects the v3 virtual environment even
if another folder's environment is accidentally active:

```powershell
.\run.ps1 `
  --source path\to\test-video.mp4 `
  --weights models\drone.pt `
  --image-size 1280 `
  --confidence 0.10 `
  --classes drone bird aircraft `
  --output runs\baseline.jsonl `
  --annotated-video runs\baseline.mp4 `
  --show `
  --preview-width 960
```

Use `--source 0` for the default camera. The low default confidence is
intentional: the later tracker will use separate acquisition and continuation
thresholds. It should not yet be treated as an alert threshold.
Progress is printed every ten frames by default. Press `Ctrl+C` to stop; the
JSONL and MP4 outputs are then closed cleanly and remain usable.
`--preview-width` changes only the display window, not detector accuracy or the
saved video's resolution. Use `--preview-width 0` to display at native size.

### Included local test clip

The development workspace may contain `videos/pexels-drone-meadow-9404253.mp4`,
a six-second 1080p ground-view sanity-check clip downloaded from Pexels. Video
binaries are ignored by Git. Its source page is:

https://www.pexels.com/video/a-drone-flying-over-a-grassy-field-9404253/

The initial POC checkpoint is the drone-fine-tuned YOLO11x `best.pt` from:

https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/tree/main/weight

Store it as `models/drone-yolo11x.pt`. Model files are ignored by Git. This is
third-party pickle-based PyTorch data, so only load it after accepting the
source and model-card terms.

## Output schema

Each JSONL row contains the source, frame index, timestamp, dimensions,
inference duration, and zero or more normalised detections:

```json
{"frame_index":0,"timestamp_seconds":0.0,"source":"clip.mp4","frame_width":1920,"frame_height":1080,"inference_ms":21.3,"detections":[{"bbox":{"x1":100.0,"y1":80.0,"x2":120.0,"y2":96.0},"confidence":0.73,"class_id":0,"class_name":"drone","model_name":"ultralytics:drone.pt","tile_origin":null}]}
```

## Evaluation principles

- Split datasets by complete recording session, never random adjacent frames.
- Report false alarms per hour and track recall as well as image-level mAP.
- Break results down by modality, camera, location and target pixel size.
- Keep research-only datasets out of weights intended for commercial use.
- Treat host-displayed licences as candidates until source-image provenance and
  exact acquisition terms have been retained.

## Planned milestones

1. Baseline detector comparison using this common output format.
2. SAHI/tiled inference for 1080p and 4K small targets.
3. Multi-frame tracking with tentative, confirmed and lost states.
4. Dataset-grounded evaluation and model comparison reports.
5. Independent RGB and thermal detectors followed by calibrated track fusion.
