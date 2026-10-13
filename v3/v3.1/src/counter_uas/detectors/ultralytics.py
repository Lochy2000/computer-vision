from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from counter_uas.core import BoundingBox, Detection, Frame
from counter_uas.detectors.base import Detector


class UltralyticsDetector(Detector):
    """Optional YOLO adapter kept outside the stable core records."""

    def __init__(
        self,
        weights: str | Path,
        *,
        confidence: float = 0.10,
        image_size: int = 640,
        device: str | None = None,
        allowed_classes: set[str] | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics support is optional. Install with: "
                "python -m pip install -e '.[ultralytics]'"
            ) from exc

        weights_path = Path(weights)
        if not weights_path.exists():
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
    def model_id(self) -> str:
        backend = "openvino" if self._weights.is_dir() else "ultralytics"
        return f"{backend}:{self._weights.name}"

    def detect(self, image: np.ndarray, frame: Frame) -> tuple[Detection, ...]:
        self.validate_image(image, frame)
        kwargs: dict[str, Any] = {
            "source": image,
            "conf": self._confidence,
            "imgsz": self._image_size,
            "verbose": False,
        }
        if self._device:
            kwargs["device"] = self._device

        result = self._model.predict(**kwargs)[0]
        if result.boxes is None:
            return ()

        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(result.names[class_id])
            if self._allowed_classes is not None and class_name.casefold() not in self._allowed_classes:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=float(box.conf[0].item()),
                    class_id=class_id,
                    class_name=class_name,
                    model_id=self.model_id,
                )
            )
        return tuple(detections)

