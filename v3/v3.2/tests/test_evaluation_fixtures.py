from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from counter_uas.evaluation import (
    FrameObservations,
    ObservationSequence,
    load_observation_sequence,
    save_observation_sequence,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "synthetic-single-target-gap.json"
)


def test_saved_fixture_replays_detection_gap_including_empty_frame() -> None:
    sequence = load_observation_sequence(FIXTURE)
    replay = tuple(sequence.replay())
    assert sequence.fixture_id == "synthetic-single-target-gap-v1"
    assert [batch.frame.sequence_number for batch in replay] == [0, 1, 2]
    assert [len(batch.observations) for batch in replay] == [1, 0, 1]


def test_fixture_round_trip_preserves_records(tmp_path: Path) -> None:
    original = load_observation_sequence(FIXTURE)
    destination = tmp_path / "round-trip.json"
    save_observation_sequence(original, destination)
    assert load_observation_sequence(destination) == original


def test_fixture_rejects_mixed_sensor_frames() -> None:
    original = load_observation_sequence(FIXTURE)
    foreign_frame = replace(original.frames[0].frame, sensor_id="south-camera")
    foreign_batch = FrameObservations(foreign_frame, ())
    with pytest.raises(ValueError, match="sensor_id"):
        replace(original, frames=(foreign_batch, *original.frames[1:]))


def test_fixture_rejects_non_increasing_frame_sequence() -> None:
    original = load_observation_sequence(FIXTURE)
    repeated_number = replace(
        original.frames[1],
        frame=replace(original.frames[1].frame, sequence_number=0),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(original, frames=(original.frames[0], repeated_number))


def test_fixture_rejects_duplicate_frame_ids() -> None:
    original = load_observation_sequence(FIXTURE)
    duplicate = replace(
        original.frames[1],
        frame=replace(original.frames[1].frame, frame_id="000000"),
    )
    with pytest.raises(ValueError, match="frame IDs"):
        replace(original, frames=(original.frames[0], duplicate))


def test_fixture_rejects_duplicate_observation_ids_across_frames() -> None:
    original = load_observation_sequence(FIXTURE)
    first_observation = original.frames[0].observations[0]
    final_batch = original.frames[2]
    duplicate_id = replace(
        final_batch.observations[0],
        observation_id=first_observation.observation_id,
    )
    with pytest.raises(ValueError, match="across the fixture"):
        replace(
            original,
            frames=(
                original.frames[0],
                original.frames[1],
                FrameObservations(final_batch.frame, (duplicate_id,)),
            ),
        )


def test_fixture_requires_at_least_one_frame() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ObservationSequence("fixture", "source", "sensor", ())

