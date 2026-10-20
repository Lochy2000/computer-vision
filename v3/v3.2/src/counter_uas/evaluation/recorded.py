from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from counter_uas.core import SensorObservation
from counter_uas.evaluation.fixtures import ObservationSequence
from counter_uas.tracking.records import LocalTrack


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_observations(sequence: ObservationSequence) -> dict[str, Any]:
    counts = Counter(
        observation.detection.class_name
        for batch in sequence.frames
        for observation in batch.observations
    )
    populated = sum(bool(batch.observations) for batch in sequence.frames)
    return {
        "frame_count": len(sequence.frames),
        "frames_with_observations": populated,
        "empty_frame_count": len(sequence.frames) - populated,
        "observation_count": sum(counts.values()),
        "class_counts": dict(sorted(counts.items())),
    }


def summarize_replay(
    sequence: ObservationSequence,
    frame_tracks: Sequence[Sequence[LocalTrack]],
) -> dict[str, Any]:
    if len(frame_tracks) != len(sequence.frames):
        raise ValueError("frame_tracks must contain one entry per fixture frame")
    track_ids = {track.track_id for tracks in frame_tracks for track in tracks}
    confirmed_ids = {
        track.track_id
        for tracks in frame_tracks
        for track in tracks
        if track.state.value == "confirmed"
    }
    return {
        **summarize_observations(sequence),
        "frames_with_any_track": sum(bool(tracks) for tracks in frame_tracks),
        "frames_with_confirmed_track": sum(
            any(track.state.value == "confirmed" for track in tracks)
            for tracks in frame_tracks
        ),
        "unique_reported_track_ids": sorted(track_ids),
        "confirmed_track_ids": sorted(confirmed_ids),
    }
