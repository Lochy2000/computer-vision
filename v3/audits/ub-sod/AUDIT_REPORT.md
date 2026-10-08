# UB-SOD training-readiness audit

## Decision

**Conditionally suitable for training; not suitable for trustworthy model
selection with its supplied splits.** The files and annotations are internally
well formed, but the train/validation/test partition contains substantial
scene-level overlap. Training should wait until the dataset is regrouped by
recording or scene and an independent test set is reserved.

## Dataset and grain

- 7,118 readable RGB images: 5,765 train, 641 validation and 712 test.
- 18,042 valid boxes: 10,047 UAV and 7,995 Bird.
- Every image has a label and every label has an image.
- Every image contains at least one labelled target; there are no pure-negative
  images.
- 181 distinct image dimensions; 4,822 images are 1920 x 1080.

## Findings

| Severity | Finding | Evidence | Training impact |
|---|---|---|---|
| High | Scene leakage across supplied splits | 886/1,353 holdout images have a dHash-near training candidate; 686 (50.7% of holdout) retain direct resized-pixel correlation >=0.95 and 536 (39.6%) exceed 0.98 | Validation/test metrics are likely optimistic and should not drive model choice |
| High | No pure-negative images | 0/7,118 labels are empty | Cannot estimate or train strongly against false alarms on empty sky, insects, clouds or unrelated aircraft-only scenes |
| Medium | Extremely small labels | 13,288/18,042 boxes (73.65%) are below 32x32 px; 816 have a minimum side below 8 px | Strong small-target relevance, but some targets may lose detail at 640 input size |
| Low | Structural annotation validity | 0 malformed class/coordinate rows, 0 unreadable images and 0 boxes outside image bounds | Suitable as source training material after split repair |
| Low | Exact duplicates | 0 SHA-256-identical images across splits | Byte deduplication alone is insufficient; sequence-level grouping is still required |

## Visual review

The fixed visual samples show plausible UAV and bird boxes, diverse weather and
airport backgrounds, and many genuinely small targets. They also visibly repeat
the same buildings, runways, wing views and camera sequences across train,
validation and test, confirming that the similarity warning is not merely a
hash collision.

## Required remediation

1. Cluster images into recording/scene groups using visual similarity, filename
   adjacency where justified, and manual review of cluster boundaries.
2. Assign whole groups to train, validation or test; never split adjacent frames.
3. Re-run exact and perceptual leakage checks on the rebuilt splits.
4. Add complete negative videos and independent external footage.
5. Use the repaired validation split for YOLO11s/YOLO11m selection and keep the
   repaired test split untouched until the final comparison.

## Evidence

- `audit.json`: complete structural and distribution profile.
- `objects.csv`: one row per labelled object.
- `near_pairs.csv`: direct pixel evidence for perceptual-hash candidate pairs.
- `small_targets.jpg`, `stratified_random.jpg`, and
  `near_duplicate_candidates.jpg`: fixed visual audit sheets.
- `UB-SOD-audit.ipynb`: executable audit narrative.
