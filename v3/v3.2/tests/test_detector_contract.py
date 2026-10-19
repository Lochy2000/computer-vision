from datetime import datetime, timezone

import numpy as np
import pytest

from counter_uas.core import BoundingBox, Detection, Frame
from counter_uas.detectors import Detector


class FakeDetector(Detector):
    @property
    def model_id(self) -> str:
        return "fake:contract-test"

    def detect(self, image: np.ndarray, frame: Frame) -> tuple[Detection, ...]:
        self.validate_image(image, frame)
        return (Detection(BoundingBox(1, 2, 5, 8), 0.75, 0, "UAV", self.model_id),)


def frame(width: int = 16, height: int = 12) -> Frame:
    return Frame(
        frame_id="sensor-1:42",
        sensor_id="sensor-1",
        sequence_number=42,
        captured_at=datetime(2026, 10, 9, tzinfo=timezone.utc),
        width=width,
        height=height,
    )


def test_detector_backend_can_be_replaced_without_yolo() -> None:
    detections = FakeDetector().detect(np.zeros((12, 16, 3), dtype=np.uint8), frame())
    assert len(detections) == 1
    assert detections[0].model_id == "fake:contract-test"


def test_detector_rejects_image_metadata_mismatch() -> None:
    with pytest.raises(ValueError, match="do not match"):
        FakeDetector().detect(np.zeros((10, 10, 3), dtype=np.uint8), frame())

