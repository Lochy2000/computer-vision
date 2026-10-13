from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from counter_uas.core.detections import Detection, SCHEMA_VERSION
from counter_uas.core.identifiers import require_identifier


def _require_aware_timestamp(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("captured_at must include a timezone")


@dataclass(frozen=True, slots=True)
class Frame:
    """Metadata identifying one captured image without storing its pixels."""

    frame_id: str
    sensor_id: str
    sequence_number: int
    captured_at: datetime
    width: int
    height: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_identifier(self.frame_id, "frame_id")
        require_identifier(self.sensor_id, "sensor_id")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        _require_aware_timestamp(self.captured_at)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported Frame schema: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["captured_at"] = self.captured_at.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class SensorObservation:
    """A timestamped sensor envelope around one model detection.

    Bearing and elevation remain optional until a later calibrated-camera stage.
    This record does not represent a track or threat assessment.
    """

    observation_id: str
    frame_id: str
    sensor_id: str
    captured_at: datetime
    detection: Detection
    bearing_degrees: float | None = None
    elevation_degrees: float | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_identifier(self.observation_id, "observation_id")
        require_identifier(self.frame_id, "frame_id")
        require_identifier(self.sensor_id, "sensor_id")
        _require_aware_timestamp(self.captured_at)
        if self.bearing_degrees is not None and not -180.0 <= self.bearing_degrees <= 180.0:
            raise ValueError("bearing_degrees must be between -180 and 180")
        if self.elevation_degrees is not None and not -90.0 <= self.elevation_degrees <= 90.0:
            raise ValueError("elevation_degrees must be between -90 and 90")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SensorObservation schema: {self.schema_version}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["captured_at"] = self.captured_at.isoformat()
        return payload

