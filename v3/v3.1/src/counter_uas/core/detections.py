from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in full-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(isfinite(value) for value in coordinates):
            raise ValueError("Bounding-box coordinates must be finite")
        if min(coordinates) < 0:
            raise ValueError("Bounding-box coordinates must be non-negative")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Detection:
    """A model result with no tracking or threat interpretation."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    model_id: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not self.class_name.strip():
            raise ValueError("class_name must not be empty")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported Detection schema: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

