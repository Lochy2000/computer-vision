# Counter-UAS v3.2

`v3.2` is the tracking evaluation and implementation line. It starts from a
code snapshot of the tested v3.1 contracts; it does not import v3.1 at runtime.

One folder contains the complete v3.2 line. Incremental checkpoints are named
`3.2.1`, `3.2.2`, `3.2.3`, and so on; they do not receive separate project
folders.

## Current checkpoint: 3.2.1

This checkpoint contains:

- the carried-forward detector, observation, and tracker contracts;
- a versioned saved-observation fixture format;
- a replay reader that includes empty frames;
- validation that prevents mixed sensors, unordered frames, duplicate frames,
  and mismatched observations;
- simple, explicitly defined single-target identity metrics;
- a small synthetic detector-gap fixture and unit tests.

It contains no ByteTrack, BoT-SORT, production tracking algorithm, video
pipeline, event database, thermal processing, fusion, or alerting.

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

