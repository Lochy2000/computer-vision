from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from counter_uas.core.detections import SCHEMA_VERSION
from counter_uas.core.identifiers import require_identifier
from counter_uas.core.observations import SensorObservation


class TrackState(str, Enum):
    """Implementation-independent lifecycle exposed to downstream modules."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


def _require_aware_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


@dataclass(frozen=True, slots=True)
class LocalTrack:
    """Bounded snapshot of one object track produced by a single sensor.

    Full observation history belongs in a later event-log module. This record
    intentionally carries only the latest observation and summary counters.
    """

    track_id: str
    tracker_id: str
    sensor_id: str
    state: TrackState
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int
    missed_updates: int
    latest_observation: SensorObservation
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_identifier(self.track_id, "track_id")
        require_identifier(self.tracker_id, "tracker_id")
        require_identifier(self.sensor_id, "sensor_id")
        if not isinstance(self.state, TrackState):
            raise ValueError("state must be a TrackState value")
        _require_aware_timestamp(self.first_seen_at, "first_seen_at")
        _require_aware_timestamp(self.last_seen_at, "last_seen_at")
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at must not be earlier than first_seen_at")
        if self.observation_count < 1:
            raise ValueError("observation_count must be at least 1")
        if self.missed_updates < 0:
            raise ValueError("missed_updates must be non-negative")
        if self.latest_observation.sensor_id != self.sensor_id:
            raise ValueError("latest observation belongs to a different sensor")
        if self.latest_observation.captured_at != self.last_seen_at:
            raise ValueError("latest observation timestamp must equal last_seen_at")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported LocalTrack schema: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "tracker_id": self.tracker_id,
            "sensor_id": self.sensor_id,
            "state": self.state.value,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "observation_count": self.observation_count,
            "missed_updates": self.missed_updates,
            "latest_observation": self.latest_observation.to_dict(),
            "schema_version": self.schema_version,
        }

