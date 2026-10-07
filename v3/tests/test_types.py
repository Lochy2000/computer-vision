import pytest

from drone_detection.types import BoundingBox, Detection, FrameResult


def test_bounding_box_geometry() -> None:
    box = BoundingBox(10, 20, 30, 50)
    assert box.width == 20
    assert box.height == 30
    assert box.area == 600


def test_invalid_bounding_box_is_rejected() -> None:
    with pytest.raises(ValueError):
        BoundingBox(10, 20, 5, 50)


def test_frame_result_serialises_nested_detections() -> None:
    detection = Detection(BoundingBox(1, 2, 3, 4), 0.75, 0, "drone", "test")
    result = FrameResult(1, 0.04, "clip.mp4", 1920, 1080, 12.5, (detection,))
    payload = result.to_dict()
    assert payload["detections"][0]["class_name"] == "drone"
    assert payload["detections"][0]["bbox"]["x1"] == 1

