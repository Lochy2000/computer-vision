# First controlled training experiment

No training has started. The selected first candidate dataset is UB-SOD, and
the existing third-party YOLO11x checkpoint remains an untouched POC baseline.

## Model order

1. Fine-tune YOLO26n from the standard pretrained base weights.
2. Fine-tune YOLO26s with the identical data, seed, resolution and schedule if
   the nano model does not meet the accuracy target.
3. Optionally fine-tune YOLO11n as a generation-control baseline.
4. Compare the selected model with the current YOLO11x POC on the locked UB-SOD test split
   and on independent full-length negative videos.

YOLO26n comes first because it is the smallest current edge candidate and has
official Raspberry Pi deployment support. YOLO26s is the controlled
accuracy/capacity comparison. Training an `m` or `x` model first would spend
substantially more compute without answering the edge question.

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

- Models: `yolo26n.pt`, then `yolo26s.pt` if justified by validation results
- Task: two-class object detection (`UAV`, `Bird`)
- Image size: begin at 640 for the controlled comparison
- Test data: publisher-provided test split, evaluated only after selection
- Deployment exports: derived after training; OpenVINO for this laptop and an
  edge runtime such as NCNN for a Raspberry Pi candidate

Epoch count, batch size and augmentation are intentionally not fixed until the
images and available training hardware have been audited.

## Audit gate status

The structural and visual audit completed on 2026-10-08. Annotation validity
passed, but the supplied image-level splits failed the scene-independence gate.
The derived `splits/ub-sod-scene-v1` partition keeps inferred visual components
intact and passed its automated and visual re-audit. It is approved for the
first controlled YOLO26n/YOLO26s development comparison. External videos are
still required for final field-performance claims. Evidence is in
`audits/ub-sod/AUDIT_REPORT.md` and
`audits/ub-sod-scene-v1/AUDIT_REPORT.md`.

## Reproducible preflight and training

The command below validates all 7,118 image/label pairs, the two-class mapping,
the frozen split counts and the absence of filename overlap. It writes a
machine-specific generated dataset YAML and a JSON record, but does not
download a model or start training:

```powershell
python tools\train_ubsod.py --preflight-only
```

After copying the repository and `data/` directory to an NVIDIA training
machine or Colab runtime, run the same preflight there. The first training run
is then:

```powershell
python tools\train_ubsod.py --device 0
```

The script starts from generic pretrained `yolo26n.pt` weights and fine-tunes
two detection classes: `UAV` and `Bird`. It never includes the locked test list
in the training YAML. Do not run this full command on the current CPU-only
laptop unless a deliberately slow CPU experiment is intended.
