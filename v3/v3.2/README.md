# Counter-UAS v3.2

`v3.2` is the tracking evaluation and implementation line. It starts from a
code snapshot of the tested v3.1 contracts; it does not import v3.1 at runtime.

One folder contains the complete v3.2 line. Incremental checkpoints are named
`3.2.1`, `3.2.2`, `3.2.3`, and so on; they do not receive separate project
folders.

## Current checkpoint: 3.2.4

This checkpoint contains:

- the carried-forward detector, observation, and tracker contracts;
- a versioned saved-observation fixture format;
- a replay reader that includes empty frames;
- validation that prevents mixed sensors, unordered frames, duplicate frames,
  and mismatched observations;
- simple, explicitly defined single-target identity metrics;
- a small synthetic detector-gap fixture and unit tests.
- a pinned Ultralytics ByteTrack adapter behind the `Tracker` interface;
- synthetic lifecycle checks covering tentative, lost, recovered, and reset
  behaviour.
- a frozen 61-frame recorded-video observation fixture;
- a hash-bound detector manifest, ByteTrack replay, manual review, and visual
  observation overlay.
- a detector-validation policy and automated readiness audit covering file
  hashes, annotations, splits, duplicate content, and required scenario strata.

It contains no BoT-SORT comparison, live video pipeline, event database,
thermal processing, fusion, or alerting.

The 3.2.2 baseline tracks only class ID `0` (`UAV`) and uses the configuration
in `configs/bytetrack-3.2.2.json`. It consumes the saved observation contract;
it does not call Ultralytics' combined detector/tracker pipeline.

The first controlled real-video case is deliberately retained as a failed
baseline. The visible drone received fragmented component/background boxes and
ByteTrack produced no track. See
`checkpoints/3.2.3-recorded-video-evaluation.md` before changing thresholds.

The 3.2.4 audit reuses the 12 existing close-drone candidate images. Their
files and hashes are valid, but they are not annotated and five required
scenario strata are absent, so the development validation gate is correctly
closed. See `checkpoints/3.2.4-detector-validation-contract.md`.

## Evaluation flow

```text
Frozen detector observations
            ↓
ObservationSequence fixture
            ↓
Replay the identical frames to each tracker
            ↓
Compare identity assignments with the same metric definitions
```

The detector is run once when a real fixture is prepared. Candidate trackers
then receive the same observations, including frames with no detections. This
keeps detector variation out of the tracker comparison.

## Development check

From this directory:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Large weights, datasets, and videos remain in the parent v3 project and are not
duplicated in v3.2.

