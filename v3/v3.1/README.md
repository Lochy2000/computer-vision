# Counter-UAS v3.1

This is the staged, modular successor to the frozen `v3` detector baseline.

## Current stage: 3.1-C

This stage contains only:

- versioned core records for frames, detections, and sensor observations;
- a detector interface that is independent of YOLO;
- an optional Ultralytics YOLO adapter;
- unit tests for the contracts and interface.
- a repeatable compatibility check between the frozen v3 detector and the v3.1
  adapter.
- deterministic construction and JSON serialization of sensor observations.

It does **not** contain tracking, event persistence, multi-node fusion, camera
geometry, thermal fusion, perimeter alerts, or SAPIENT integration.

## Detector compatibility checkpoint

The migration verifier runs both adapters with the same checkpoint, decoded
frames, image size, confidence threshold, device, and class filter. It compares
detection counts, classes, confidence values, and box coordinates. Timing is not
part of this check.

The recorded 13 October 2026 checkpoint passed on all five selected frames with
four detections from each adapter and no output differences. See
`checkpoints/3.1-B-detector-compatibility.md` for the decision and limitations,
and `verification/detector-compatibility.json` for the machine-readable record.

From `v3/v3.1`, the recorded check can be reproduced with:

```powershell
..\.venv\Scripts\python.exe tools\verify_detector_compatibility.py `
  --source "..\videos\pexels-drone-meadow-9404253.mp4" `
  --weights "..\runs\colab-imports\ubsod-scene-v1-yolo26n-640-seed42\weights\best.pt" `
  --output "verification\detector-compatibility.json" `
  --frame-indices "0,30,60,90,120" `
  --image-size 640 `
  --confidence 0.10 `
  --device cpu
```

## Module boundary

```text
image + Frame -> Detector -> Detection -> SensorObservation
```

The raw image is deliberately not stored inside `Frame` or
`SensorObservation`. Images/video can be managed by a later evidence component
without coupling all records to large pixel arrays.

Observation construction is a pure operation: it copies frame context around
each ordered detection and assigns a replay-stable event ID. It does not create
a track. See `checkpoints/3.1-C-observation-construction.md` for the precise
contract and acceptance checks.

## Development check

From this directory:

```powershell
python -m pytest -q
```

Install the package for development with:

```powershell
python -m pip install -e ".[dev]"
```

YOLO support remains optional:

```powershell
python -m pip install -e ".[ultralytics,dev]"
```

## Compatibility policy

Core records carry `schema_version = "1.0"`. Later stages may add optional
fields without breaking this contract. Breaking changes require a new schema
version and migration or adapter tests.

