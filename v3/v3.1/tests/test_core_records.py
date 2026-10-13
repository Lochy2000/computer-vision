from datetime import datetime, timezone

import pytest

from counter_uas.core import BoundingBox, Detection, Frame, SensorObservation


NOW = datetime(2026, 10, 9, 8, 30, tzinfo=timezone.utc)


def example_detection() -> Detection:
    return Detection(
        bbox=BoundingBox(10, 20, 30, 50),
        confidence=0.8,
        class_id=0,
        class_name="UAV",
        model_id="yolo26n:baseline-a",
    )


def test_bounding_box_geometry() -> None:
    box = BoundingBox(10, 20, 30, 50)
    assert box.width == 20
    assert box.height == 30
    assert box.area == 600


@pytest.mark.parametrize(
    "coordinates",
    [(-1, 0, 10, 10), (10, 0, 10, 10), (0, 10, 10, 5)],
)
def test_invalid_bounding_box_is_rejected(coordinates: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        BoundingBox(*coordinates)


def test_frame_requires_timezone_and_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="timezone"):
        Frame("frame-1", "sensor-1", 0, datetime(2026, 10, 9), 640, 480)
    with pytest.raises(ValueError, match="dimensions"):
        Frame("frame-1", "sensor-1", 0, NOW, 0, 480)


def test_observation_serialises_as_stable_json_ready_record() -> None:
    observation = SensorObservation(
        observation_id="observation-1",
        frame_id="frame-1",
        sensor_id="sensor-1",
        captured_at=NOW,
        detection=example_detection(),
    )
    payload = observation.to_dict()
    assert payload["schema_version"] == "1.0"
    assert payload["captured_at"] == "2026-10-09T08:30:00+00:00"
    assert payload["detection"]["class_name"] == "UAV"
    assert payload["bearing_degrees"] is None


def test_observation_rejects_invalid_angles() -> None:
    with pytest.raises(ValueError, match="bearing"):
        SensorObservation("o", "f", "s", NOW, example_detection(), 181.0)

