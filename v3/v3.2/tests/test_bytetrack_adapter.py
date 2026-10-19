from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from counter_uas.core import BoundingBox, Detection
from counter_uas.evaluation import load_observation_sequence
from counter_uas.observations import build_observations
from counter_uas.tracking import ByteTrackConfig, ByteTrackTracker, TrackState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "bytetrack-3.2.2.json"
FIXTURE = PROJECT_ROOT / "fixtures" / "synthetic-single-target-gap.json"


def test_pinned_configuration_loads_expected_uav_only_baseline() -> None:
    config = ByteTrackConfig.from_json(CONFIG)
    assert config.tracker_id == "ultralytics-bytetrack:3.2.2-v1"
    assert config.tracked_class_ids == (0,)
    assert config.track_buffer_frames == 30
    assert config.minimum_confirmed_observations == 2


def test_synthetic_gap_preserves_identity_and_lifecycle() -> None:
    tracker = ByteTrackTracker(ByteTrackConfig.from_json(CONFIG))
    batches = tuple(load_observation_sequence(FIXTURE).replay())

    first = tracker.update(batches[0].frame, batches[0].observations)
    gap = tracker.update(batches[1].frame, batches[1].observations)
    recovered = tracker.update(batches[2].frame, batches[2].observations)

    assert len(first) == len(gap) == len(recovered) == 1
    assert first[0].track_id == gap[0].track_id == recovered[0].track_id
    assert first[0].state is TrackState.TENTATIVE
    assert first[0].observation_count == 1
    assert gap[0].state is TrackState.LOST
    assert gap[0].missed_updates == 1
    assert gap[0].latest_observation == first[0].latest_observation
    assert recovered[0].state is TrackState.CONFIRMED
    assert recovered[0].observation_count == 2
    assert recovered[0].missed_updates == 0


def test_reset_makes_fixture_replay_deterministic() -> None:
    tracker = ByteTrackTracker(ByteTrackConfig.from_json(CONFIG))
    first_batch = load_observation_sequence(FIXTURE).frames[0]
    first_id = tracker.update(
        first_batch.frame,
        first_batch.observations,
    )[0].track_id
    tracker.reset()
    replayed_id = tracker.update(
        first_batch.frame,
        first_batch.observations,
    )[0].track_id
    assert replayed_id == first_id


def test_adapter_rejects_repeated_or_reversed_frame_sequence() -> None:
    tracker = ByteTrackTracker(ByteTrackConfig.from_json(CONFIG))
    batch = load_observation_sequence(FIXTURE).frames[0]
    tracker.update(batch.frame, batch.observations)
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update(batch.frame, batch.observations)


def test_uav_only_baseline_ignores_bird_observation() -> None:
    tracker = ByteTrackTracker(ByteTrackConfig.from_json(CONFIG))
    batch = load_observation_sequence(FIXTURE).frames[0]
    bird_detection = Detection(
        bbox=BoundingBox(100, 200, 130, 225),
        confidence=0.9,
        class_id=1,
        class_name="Bird",
        model_id="synthetic:detector",
    )
    bird_observation = build_observations(batch.frame, (bird_detection,))
    assert tracker.update(batch.frame, bird_observation) == ()


def test_adapter_rejects_mixed_sensor_streams() -> None:
    tracker = ByteTrackTracker(ByteTrackConfig.from_json(CONFIG))
    batches = load_observation_sequence(FIXTURE).frames
    tracker.update(batches[0].frame, batches[0].observations)
    foreign_frame = replace(
        batches[1].frame,
        sensor_id="south-camera",
    )
    with pytest.raises(ValueError, match="cannot mix"):
        tracker.update(foreign_frame, ())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("track_low_threshold", 1.1, "between 0 and 1"),
        ("track_buffer_frames", 0, "at least 1"),
        ("minimum_confirmed_observations", 0, "at least 1"),
        ("tracked_class_ids", (), "must not be empty"),
    ],
)
def test_invalid_configuration_is_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ByteTrackConfig(**{field: value})  # type: ignore[arg-type]

