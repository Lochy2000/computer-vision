from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import pytest

from counter_uas.core import BoundingBox, Detection, Frame, SensorObservation
from counter_uas.observations import build_observations
from counter_uas.tracking import LocalTrack, Tracker, TrackState


NOW = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)


def make_frame(
    *,
    frame_id: str = "42",
    sensor_id: str = "north-camera",
    captured_at: datetime = NOW,
) -> Frame:
    return Frame(frame_id, sensor_id, 42, captured_at, 640, 480)


def make_detection() -> Detection:
    return Detection(
        BoundingBox(10, 20, 30, 40),
        0.8,
        0,
        "UAV",
        "yolo26n:baseline-a",
    )


def make_observation(frame: Frame | None = None) -> SensorObservation:
    source_frame = frame or make_frame()
    return build_observations(source_frame, (make_detection(),))[0]


class FakeTracker(Tracker):
    """Contract test double; not a production association algorithm."""

    def __init__(self) -> None:
        self.update_count = 0

    @property
    def tracker_id(self) -> str:
        return "fake:contract-test"

    def update(
        self,
        frame: Frame,
        observations: Sequence[SensorObservation],
    ) -> tuple[LocalTrack, ...]:
        self.validate_update(frame, observations)
        self.update_count += 1
        return tuple(
            LocalTrack(
                track_id=f"{frame.sensor_id}:track:{index}",
                tracker_id=self.tracker_id,
                sensor_id=frame.sensor_id,
                state=TrackState.TENTATIVE,
                first_seen_at=observation.captured_at,
                last_seen_at=observation.captured_at,
                observation_count=1,
                missed_updates=0,
                latest_observation=observation,
            )
            for index, observation in enumerate(observations)
        )

    def reset(self) -> None:
        self.update_count = 0


def test_tracker_backend_can_be_replaced_without_an_algorithm_dependency() -> None:
    frame = make_frame()
    tracks = FakeTracker().update(frame, (make_observation(frame),))
    assert len(tracks) == 1
    assert tracks[0].tracker_id == "fake:contract-test"
    assert tracks[0].state is TrackState.TENTATIVE


def test_empty_frame_update_is_valid_and_reset_is_explicit() -> None:
    tracker = FakeTracker()
    assert tracker.update(make_frame(), ()) == ()
    assert tracker.update_count == 1
    tracker.reset()
    assert tracker.update_count == 0


@pytest.mark.parametrize(
    "observation_frame",
    [
        make_frame(sensor_id="south-camera"),
        make_frame(frame_id="43"),
        make_frame(captured_at=NOW + timedelta(milliseconds=1)),
    ],
)
def test_tracker_rejects_observations_from_another_frame(
    observation_frame: Frame,
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        FakeTracker().update(make_frame(), (make_observation(observation_frame),))


def test_tracker_rejects_duplicate_observation_ids() -> None:
    frame = make_frame()
    observation = make_observation(frame)
    with pytest.raises(ValueError, match="unique"):
        FakeTracker().update(frame, (observation, observation))


def test_track_serialisation_is_json_ready_and_bounded() -> None:
    observation = make_observation()
    track = LocalTrack(
        track_id="north-camera:track:1",
        tracker_id="fake:contract-test",
        sensor_id="north-camera",
        state=TrackState.CONFIRMED,
        first_seen_at=NOW - timedelta(seconds=1),
        last_seen_at=NOW,
        observation_count=3,
        missed_updates=0,
        latest_observation=observation,
    )
    payload = track.to_dict()
    assert payload["state"] == "confirmed"
    assert payload["first_seen_at"] == "2026-10-15T08:59:59+00:00"
    assert payload["latest_observation"]["observation_id"].endswith(
        ":observation:0000"
    )
    assert "observation_history" not in payload


def test_track_rejects_invalid_time_order() -> None:
    with pytest.raises(ValueError, match="earlier"):
        LocalTrack(
            "track-1",
            "tracker-1",
            "north-camera",
            TrackState.TENTATIVE,
            NOW + timedelta(seconds=1),
            NOW,
            1,
            0,
            make_observation(),
        )


def test_track_rejects_cross_sensor_latest_observation() -> None:
    south_frame = make_frame(sensor_id="south-camera")
    with pytest.raises(ValueError, match="different sensor"):
        LocalTrack(
            "track-1",
            "tracker-1",
            "north-camera",
            TrackState.TENTATIVE,
            NOW,
            NOW,
            1,
            0,
            make_observation(south_frame),
        )


def test_track_rejects_invalid_counters() -> None:
    with pytest.raises(ValueError, match="observation_count"):
        LocalTrack(
            "track-1",
            "tracker-1",
            "north-camera",
            TrackState.TENTATIVE,
            NOW,
            NOW,
            0,
            0,
            make_observation(),
        )


def test_track_rejects_untyped_lifecycle_state() -> None:
    with pytest.raises(ValueError, match="TrackState"):
        LocalTrack(
            "track-1",
            "tracker-1",
            "north-camera",
            "confirmed",  # type: ignore[arg-type]
            NOW,
            NOW,
            1,
            0,
            make_observation(),
        )

