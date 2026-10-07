from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from drone_detection.detectors.base import Detector
from drone_detection.types import BoundingBox, Detection


class UltralyticsDetector(Detector):
    """Adapter for local Ultralytics-compatible YOLO weight files."""

    def __init__(
        self,
        weights: str | Path,
        confidence: float = 0.10,
        image_size: int = 1280,
        device: str | None = None,
        allowed_classes: set[str] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics support is optional. Install with: "
                "pip install -e '.[ultralytics]'"
            ) from exc

        weights_path = Path(weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        self._model = YOLO(str(weights_path))
        self._weights = weights_path
        self._confidence = confidence
        self._image_size = image_size
        self._device = device
        self._allowed_classes = (
            {name.casefold() for name in allowed_classes} if allowed_classes else None
        )

    @property
    def name(self) -> str:
        return f"ultralytics:{self._weights.name}"

    def predict(self, frame: np.ndarray) -> list[Detection]:
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self._confidence,
            "imgsz": self._image_size,
            "verbose": False,
        }
        if self._device:
            kwargs["device"] = self._device

        result = self._model.predict(**kwargs)[0]
        names = result.names
        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(names[class_id])
            if (
                self._allowed_classes is not None
                and class_name.casefold() not in self._allowed_classes
            ):
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=float(box.conf[0].item()),
                    class_id=class_id,
                    class_name=class_name,
                    model_name=self.name,
                )
            )
        return detections

