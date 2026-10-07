from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from drone_detection.types import Detection


class Detector(ABC):
    """Common interface implemented by every detection backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Return detections in full-frame pixel coordinates."""
        raise NotImplementedError

