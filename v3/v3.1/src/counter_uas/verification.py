from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class BoxLike(Protocol):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionLike(Protocol):
    bbox: BoxLike
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class DetectionDifference:
    detection_index: int | None
    field: str
    legacy_value: object
    candidate_value: object

    def to_dict(self) -> dict[str, object]:
        return {
            "detection_index": self.detection_index,
            "field": self.field,
            "legacy_value": self.legacy_value,
            "candidate_value": self.candidate_value,
        }


def compare_detections(
    legacy: Sequence[DetectionLike],
    candidate: Sequence[DetectionLike],
    *,
    absolute_tolerance: float = 1e-5,
) -> tuple[DetectionDifference, ...]:
    """Compare ordered detector outputs while ignoring adapter-specific metadata."""
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative")

    differences: list[DetectionDifference] = []
    if len(legacy) != len(candidate):
        differences.append(
            DetectionDifference(None, "count", len(legacy), len(candidate))
        )

    for index, (old, new) in enumerate(zip(legacy, candidate, strict=False)):
        for field in ("class_id", "class_name"):
            old_value = getattr(old, field)
            new_value = getattr(new, field)
            if old_value != new_value:
                differences.append(
                    DetectionDifference(index, field, old_value, new_value)
                )

        if abs(old.confidence - new.confidence) > absolute_tolerance:
            differences.append(
                DetectionDifference(
                    index, "confidence", old.confidence, new.confidence
                )
            )

        for coordinate in ("x1", "y1", "x2", "y2"):
            old_value = getattr(old.bbox, coordinate)
            new_value = getattr(new.bbox, coordinate)
            if abs(old_value - new_value) > absolute_tolerance:
                differences.append(
                    DetectionDifference(
                        index, f"bbox.{coordinate}", old_value, new_value
                    )
                )

    return tuple(differences)

