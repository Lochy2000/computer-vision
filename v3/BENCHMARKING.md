# Portable benchmarking contract

The benchmark is model- and device-independent. A trained model is the logical
asset; PyTorch, OpenVINO, ONNX, NCNN, TFLite and TensorRT files are deployment
exports derived from it. Device-specific exports do not change the training
data or make the learned detector specific to that device.

## What is preserved

- Model identity, source, hashes and unresolved licence status in
  `manifests/models.json`.
- Test-media identity, source, hash and intended role in
  `manifests/tests.json`.
- Per-run hardware, platform, package versions, inference settings, cold-start
  time, steady-state latency, FPS and detection counts in JSON benchmark files.
- Existing JSONL detections and annotated videos remain useful for visual
  regression checks, but are ignored by Git because they can be regenerated.

The model and video binaries are ignored by Git. The manifests preserve their
identity, but durable backup of the binaries requires artifact storage or a
release archive; a Git commit alone is not a backup.

## Run a reproducible benchmark

Intel GPU example:

```powershell
.\.venv\Scripts\python.exe -m drone_detection.benchmark `
  --model-id third-party-yolo11x-drone-openvino-fp16 `
  --test-id pexels-meadow-close-drone-v1 `
  --source videos\pexels-drone-meadow-9404253.mp4 `
  --weights models\drone-yolo11x_openvino_model `
  --device intel:gpu `
  --image-size 640 `
  --confidence 0.10 `
  --classes drone `
  --max-frames 20 `
  --output benchmarks\intel-iris-xe\yolo11x-openvino-pexels-20.json
```

Run the same logical model, settings and test ID on another machine. The JSON
records retain enough context to compare sustained detector latency without
confusing first-frame compilation with normal operation.

## Test-suite maturity

The current Pexels clip is only a positive smoke test. It has no ground-truth
boxes and cannot measure precision, recall, false alarms or long-range ability.
Before selecting a production model, add a locked evaluation suite containing:

- Close, medium and genuinely distant drones, stratified by target pixel size.
- Birds, aircraft, insects, empty sky and cluttered backgrounds.
- Day, dusk, cloud, haze and adverse weather.
- Complete negative videos for false alarms per hour.
- RGB now and an independently labelled thermal partition later.
- Bounding-box annotations and video/session-level train/validation/test splits.

Do not tune thresholds on the locked test suite. Use a separate validation set.

## Prepare evaluation candidates

Extracting frames is deliberately separate from annotation. Model predictions
must not be copied into ground truth, because evaluating against a model's own
labels would produce circular and misleading results.

```powershell
.\.venv\Scripts\python.exe -m drone_detection.evalset extract `
  --source videos\pexels-drone-meadow-9404253.mp4 `
  --output-dir data\evaluation\candidates\pexels-meadow `
  --manifest manifests\evaluation-pexels-meadow.json `
  --samples 12 `
  --source-id pexels-meadow-close-drone-v1
```

The tracked manifest records source and image hashes. Extracted images live
under ignored `data/`; archive them with the model/video artifacts. Each record
remains `pending` until a person draws or verifies the drone bounding box.
After annotation, set `annotation_status` to `complete`, set `label` to its YOLO
text-file path, and validate the set:

```powershell
.\.venv\Scripts\python.exe -m drone_detection.evalset validate `
  --manifest manifests\evaluation-pexels-meadow.json
```

An empty YOLO label file is valid only for a human-reviewed negative image.

## Deployment exports

| Target | Candidate runtime | Expected artifact |
|---|---|---|
| Intel laptop/mini PC | OpenVINO | OpenVINO IR directory |
| Raspberry Pi CPU | NCNN, TFLite or ONNX Runtime | target-specific export |
| Raspberry Pi + Hailo/Coral | vendor compiler/runtime | accelerator-specific model |
| NVIDIA Jetson | TensorRT | `.engine` built on target |
| Development/reference | PyTorch | `.pt` |

The same test suite and reporting schema should be used everywhere. Accuracy
must be rechecked after quantisation or conversion; matching architecture names
do not guarantee numerically identical outputs.
