# First controlled training experiment

No training has started. The selected first candidate dataset is UB-SOD, and
the existing third-party YOLO11x checkpoint remains an untouched POC baseline.

## Model order

1. Fine-tune YOLO11s from the standard pretrained base weights.
2. Fine-tune YOLO11m with the identical data, seed, resolution and schedule.
3. Compare both with the current YOLO11x POC on the locked UB-SOD test split
   and on independent full-length negative videos.

YOLO11s comes first because it is the more plausible CPU/edge deployment
candidate. YOLO11m is the controlled accuracy/capacity comparison. Training an
`x` model first would spend substantially more compute without answering the
edge question.

## Gates before training

- Download and SHA-256 verify `UB-SOD_images.zip` from the URL recorded in
  `manifests/datasets.json`.
- Run a visual sample audit across every split and both classes.
- Confirm there are no exact or near-duplicate images across splits.
- Preserve the supplied test split; never tune confidence or epochs on it.
- Record the exact Ultralytics version, base-weight hash, seed and arguments.
- Decide how to meet Ultralytics licensing requirements for the intended use.

The laptop can run inference benchmarks, but model training should be treated
as a separate reproducible job. A cloud/NVIDIA training run will not make the
resulting weights specific to that hardware.

## Initial experiment constants

- Models: `yolo11s.pt`, then `yolo11m.pt`
- Task: two-class object detection (`UAV`, `Bird`)
- Image size: begin at 640 for the controlled comparison
- Test data: publisher-provided test split, evaluated only after selection
- Deployment exports: derived after training; OpenVINO for this laptop and an
  edge runtime such as NCNN for a Raspberry Pi candidate

Epoch count, batch size and augmentation are intentionally not fixed until the
images and available training hardware have been audited.
