# Counter-UAS V0.1 project goal

## North-star goal

Demonstrate, in a representative secure-site environment, that two or preferably
three fixed low-cost sensor nodes can detect and maintain a track on a cooperative
consumer drone, combine timestamped observations into an approximate trajectory,
predict a crossing of a mapped protected perimeter, preserve an auditable event
record, and emit a basic SAPIENT-compatible track message.

The demonstration must report detection range, time to first alert, track
continuity, false UAV alerts per hour, localization error against drone telemetry,
compute load, and power use. Exact acceptance thresholds must be agreed before the
field trial from the competition requirements and operational need.

## Current position

The preserved YOLO26n UAV/Bird checkpoint is **Baseline A**. It proves that the
project has a reproducible RGB image-detector component. It does not yet prove
long-range field detection, persistent tracking, thermal operation, multi-node
association, triangulation, perimeter warning, or a deployable prison system.

## Active milestone

Build and evaluate the one-node evidence loop:

`camera -> detector -> multi-frame tracker -> timestamped observation -> event log`

Use independent positive and negative video to measure track continuity, time to
first detection, false UAV alerts per hour, and end-to-end performance. Do not
start another broad training run until this evidence identifies a detector failure
that training can plausibly fix.

## Stage gates

1. Preserve Baseline A and never tune against its locked test.
2. Prove the one-node evidence loop.
3. Add a synchronized second node and validate association/triangulation against
   drone telemetry.
4. Add the third node, site model, perimeter prediction, and operator alert.
5. Add thermal confirmation and the SAPIENT adapter.
6. Train or compare another model only from measured failure cases.

The detailed rationale and evidence are in
`reports/project-direction/counter-uas-v0.1-direction.html`.
