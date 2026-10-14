from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from counter_uas.core import BoundingBox, Detection, Frame
from counter_uas.observations import build_observations, observations_to_json


CAPTURED_AT = datetime(2026, 10, 14, 9, 30, tzinfo=timezone.utc)


def frame() -> Frame:
    return Frame(
        frame_id="000042",
        sensor_id="north-camera",
        sequence_number=42,
        captured_at=CAPTURED_AT,
        width=1920,
        height=1080,
    )


def detection(
    class_name: str = "UAV",
    bbox: BoundingBox = BoundingBox(100, 200, 140, 230),
) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=0.75,
        class_id=0,
        class_name=class_name,
        model_id="yolo26n:baseline-a",
    )


def test_builder_propagates_frame_context_and_preserves_order() -> None:
    observations = build_observations(
        frame(),
        (detection("UAV"), detection("Bird")),
    )

    assert [item.observation_id for item in observations] == [
        "north-camera:000042:observation:0000",
        "north-camera:000042:observation:0001",
    ]
    assert [item.detection.class_name for item in observations] == ["UAV", "Bird"]
    assert all(item.frame_id == "000042" for item in observations)
    assert all(item.sensor_id == "north-camera" for item in observations)
    assert all(item.captured_at == CAPTURED_AT for item in observations)
    assert all(item.bearing_degrees is None for item in observations)


def test_replay_is_deterministic() -> None:
    detections = (detection(), detection("Bird"))
    assert build_observations(frame(), detections) == build_observations(
        frame(), detections
    )


def test_sensor_identity_prevents_cross_node_id_collision() -> None:
    north = frame()
    south = Frame(
        frame_id=north.frame_id,
        sensor_id="south-camera",
        sequence_number=north.sequence_number,
        captured_at=north.captured_at,
        width=north.width,
        height=north.height,
    )
    north_id = build_observations(north, (detection(),))[0].observation_id
    south_id = build_observations(south, (detection(),))[0].observation_id
    assert north_id != south_id


def test_json_is_stable_and_round_trippable() -> None:
    observations = build_observations(frame(), (detection(),))
    first = observations_to_json(observations)
    second = observations_to_json(observations)
    assert first == second

    payload = json.loads(first)
    assert payload[0]["schema_version"] == "1.0"
    assert payload[0]["captured_at"] == "2026-10-14T09:30:00+00:00"
    assert payload[0]["detection"]["bbox"]["x1"] == 100


def test_empty_detection_sequence_produces_empty_output() -> None:
    assert build_observations(frame(), ()) == ()
    assert observations_to_json(()) == "[]"


def test_out_of_frame_detection_is_rejected() -> None:
    outside = detection(bbox=BoundingBox(1900, 100, 1930, 120))
    with pytest.raises(ValueError, match="exceeds Frame dimensions"):
        build_observations(frame(), (outside,))


def test_non_finite_box_is_rejected_at_the_core_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        BoundingBox(0, 0, float("nan"), 10)

