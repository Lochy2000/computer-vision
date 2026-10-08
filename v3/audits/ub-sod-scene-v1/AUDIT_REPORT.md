# UB-SOD scene-v1 split audit

## Decision

**Pass for a controlled YOLO11s/YOLO11m development comparison.** All images
are assigned exactly once, connected visual components remain intact, and no
cross-split pair satisfies the final leakage rule. This is materially safer
than the publisher split, but it is not a substitute for an external test set.

## Evidence

| Check | Result |
|---|---:|
| Assigned source images | 7,118 / 7,118 |
| Missing or unexpected images | 0 |
| Visual components crossing splits | 0 |
| Cross-split pairs passing the final leakage rule | 0 |
| Train images | 5,367 (75.4%) |
| Validation images | 876 (12.3%) |
| Test images | 875 (12.3%) |

Class balance is retained: training contains 7,899 UAV and 6,347 Bird boxes;
validation contains 1,062 UAV and 820 Bird boxes; test contains 1,086 UAV and
828 Bird boxes.

## Sensitivity review

A deliberately looser screen returned 705 candidates. Visual review of the
highest-scoring sheet found that it overmatches unrelated airport/runway views
because broad sky regions dominate the resized pixels. Genuine low-error pairs
found in earlier iterations were promoted into the final grouping rule and the
split was rebuilt until none remained under that rule.

## Remaining risks

- The dataset has no authoritative recording, scene or airport identifiers.
- A visual component is an inferred proxy for a recording group.
- One inferred component contains 890 images, showing substantial redundancy.
- There are no pure-negative images.
- Site independence and real-world false alarms must be measured using separate
  videos not derived from UB-SOD.

## Recommendation

Use this split for the first controlled YOLO11s run and subsequent YOLO11m
comparison. Do not report its test result as final field performance. Preserve
the test list unchanged and add independent negative and distant-drone videos
before deployment decisions.
