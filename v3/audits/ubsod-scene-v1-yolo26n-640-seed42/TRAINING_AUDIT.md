# YOLO26n UB-SOD scene-v1 training audit

## Decision

The imported Colab run is complete and internally consistent. Select
`weights/best.pt`; do not select `weights/last.pt`. The selected checkpoint was
subsequently evaluated once on the locked test split, after this training audit
and checkpoint decision were recorded.

## Reproducibility

- Model: YOLO26n pretrained base, fine-tuned for `UAV` and `Bird`
- Image size: 640
- Seed: 42
- Epochs completed: 100
- Training images: 5,367
- Validation images: 876
- Training duration: 5,317.42 seconds (1.477 hours)
- Ultralytics: 8.4.174
- PyTorch: 2.11.0+cu130
- GPU: NVIDIA L4
- Automatic batch size: 50
- Missing/corrupt images: 0

The canonical train and validation split hashes match the audited scene-v1
partition. The imported archive contains configuration, logs, plots, resolved
lists, periodic checkpoints, `best.pt`, and `last.pt`.

## Checkpoint selection

Ultralytics fitness peaked at epoch 81 (one-indexed):

| Metric | Validation value |
|---|---:|
| Precision | 0.86137 |
| Recall | 0.71177 |
| mAP50 | 0.79693 |
| mAP50-95 | 0.37177 |

The greatest mAP50 occurred at epoch 79 (0.79772), but epoch 81 maximized the
fitness dominated by mAP50-95. The post-training validation of `best.pt`
reported overall precision 0.863, recall 0.710, mAP50 0.797, and mAP50-95
0.372. UAV mAP50/mAP50-95 was 0.867/0.450; Bird was 0.728/0.294.

Training box loss continued downward through epoch 100, while validation box
and localization losses reached minima at epoch 70 and validation
classification loss reached its minimum at epoch 80. All three validation
losses were higher by epoch 100. This is mild late overfitting and supports the
epoch-81 `best.pt` selection rather than `last.pt`.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `weights/best.pt` | `9dace11fb8f36b971a295de39b6a9bd5f20d3ba51a82defbfd615a89c0b559c5` |
| `weights/last.pt` | `f059ab87261ae0cc6efce17314b79e5d6e6f6bb68710016b5aeca037798df010` |
| `args.yaml` | `235137fc659a919d1cd001209878c64247967291b0cd3d4ed1007c86cce09d35` |
| `results.csv` | `76085945d081589e6fe86ce4e55f585883de389640fa6c47cdce6202943f6da6` |
| `preflight.json` | `c2880652c1cbd6bb256a014654205b62206f7f8901967f7a41e7fcbef10716f9` |
| `dataset.generated.yaml` | `1b1d46f55878157dca5729c19352072c0670830cb4b5baf2dbda6ac089a0f91d` |

## Interpretation limits

- These are validation results, not locked-test results.
- UB-SOD is RGB airport-domain data and does not establish thermal capability.
- The confusion matrix is generated across confidence thresholds for metric
  analysis; its background column is not a field false-alert rate.
- Independent negative videos remain necessary to measure false UAV alerts per
  hour and performance on aircraft, insects, clouds, and empty sky.
- The checkpoint is subject to Ultralytics licensing and UB-SOD attribution
  requirements documented elsewhere in the project.

## Locked-test result

The one-time evaluation completed on all 875 test images and 1,914 labelled
instances. The immutable machine-readable record is
`evaluations/ubsod-scene-v1-yolo26n-640-seed42/locked-test.json`.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Overall | 0.8622 | 0.7566 | 0.8086 | 0.3853 |
| UAV | 0.8734 | 0.6986 | 0.7860 | 0.3852 |
| Bird | 0.8511 | 0.8146 | 0.8312 | 0.3854 |

The test result is close to and slightly stronger than the aggregate
validation result, providing no sign of a validation-to-test collapse under
the derived scene split. UAV recall is the main measured weakness: at the
metric sweep, roughly 30% of labelled UAV instances are not recalled. No
further model or threshold decisions should be tuned against this locked test
set.
