# Counter-UAS v3.1

This is the staged, modular successor to the frozen `v3` detector baseline.

## Current stage: 3.1-A

This stage contains only:

- versioned core records for frames, detections, and sensor observations;
- a detector interface that is independent of YOLO;
- an optional Ultralytics YOLO adapter;
- unit tests for the contracts and interface.

It does **not** contain tracking, event persistence, multi-node fusion, camera
geometry, thermal fusion, perimeter alerts, or SAPIENT integration.

## Module boundary

```text
image + Frame -> Detector -> Detection -> SensorObservation
```

The raw image is deliberately not stored inside `Frame` or
`SensorObservation`. Images/video can be managed by a later evidence component
without coupling all records to large pixel arrays.

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

