from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from counter_uas.core import Detection, Frame


class Detector(ABC):
    """Backend-neutral contract for detecting objects in one frame."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier for the model and export used for inference."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, image: np.ndarray, frame: Frame) -> tuple[Detection, ...]:
        """Return image detections only; do not track or classify threats."""
        raise NotImplementedError

    @staticmethod
    def validate_image(image: np.ndarray, frame: Frame) -> None:
        if image.ndim not in (2, 3):
            raise ValueError("image must be a two- or three-dimensional array")
        image_height, image_width = image.shape[:2]
        if (image_width, image_height) != (frame.width, frame.height):
            raise ValueError(
                "image dimensions do not match Frame metadata: "
                f"image={image_width}x{image_height}, "
                f"frame={frame.width}x{frame.height}"
            )

