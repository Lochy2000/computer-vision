from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from counter_uas.core import BoundingBox, Detection, Frame, SensorObservation
from counter_uas.core.identifiers import require_identifier


FIXTURE_SCHEMA_VERSION = 1


def _datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _frame_from_dict(payload: dict[str, Any]) -> Frame:
    return Frame(
        frame_id=payload["frame_id"],
        sensor_id=payload["sensor_id"],
        sequence_number=payload["sequence_number"],
        captured_at=_datetime(payload["captured_at"], "frame.captured_at"),
        width=payload["width"],
        height=payload["height"],
        schema_version=payload.get("schema_version", "1.0"),
    )


def _observation_from_dict(payload: dict[str, Any]) -> SensorObservation:
    detection_payload = payload["detection"]
    box_payload = detection_payload["bbox"]
    detection = Detection(
        bbox=BoundingBox(
            x1=box_payload["x1"],
            y1=box_payload["y1"],
            x2=box_payload["x2"],
            y2=box_payload["y2"],
        ),
        confidence=detection_payload["confidence"],
        class_id=detection_payload["class_id"],
        class_name=detection_payload["class_name"],
        model_id=detection_payload["model_id"],
        schema_version=detection_payload.get("schema_version", "1.0"),
    )
    return SensorObservation(
        observation_id=payload["observation_id"],
        frame_id=payload["frame_id"],
        sensor_id=payload["sensor_id"],
        captured_at=_datetime(payload["captured_at"], "observation.captured_at"),
        detection=detection,
        bearing_degrees=payload.get("bearing_degrees"),
        elevation_degrees=payload.get("elevation_degrees"),
        schema_version=payload.get("schema_version", "1.0"),
    )


@dataclass(frozen=True, slots=True)
class FrameObservations:
    frame: Frame
    observations: tuple[SensorObservation, ...]

    def __post_init__(self) -> None:
        observation_ids: set[str] = set()
        for observation in self.observations:
            if observation.observation_id in observation_ids:
                raise ValueError("observation IDs must be unique within a frame")
            observation_ids.add(observation.observation_id)
            if observation.sensor_id != self.frame.sensor_id:
                raise ValueError("observation sensor_id does not match fixture Frame")
            if observation.frame_id != self.frame.frame_id:
                raise ValueError("observation frame_id does not match fixture Frame")
            if observation.captured_at != self.frame.captured_at:
                raise ValueError("observation timestamp does not match fixture Frame")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrameObservations:
        return cls(
            frame=_frame_from_dict(payload["frame"]),
            observations=tuple(
                _observation_from_dict(item) for item in payload["observations"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservationSequence:
    fixture_id: str
    source_id: str
    sensor_id: str
    frames: tuple[FrameObservations, ...]
    schema_version: int = FIXTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_identifier(self.fixture_id, "fixture_id")
        require_identifier(self.source_id, "source_id")
        require_identifier(self.sensor_id, "sensor_id")
        if self.schema_version != FIXTURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported observation fixture schema: {self.schema_version}"
            )
        if not self.frames:
            raise ValueError("observation fixture must contain at least one frame")

        frame_ids: set[str] = set()
        observation_ids: set[str] = set()
        previous_sequence_number: int | None = None
        previous_timestamp: datetime | None = None
        for batch in self.frames:
            frame = batch.frame
            if frame.sensor_id != self.sensor_id:
                raise ValueError("all fixture frames must belong to sensor_id")
            if frame.frame_id in frame_ids:
                raise ValueError("fixture frame IDs must be unique")
            frame_ids.add(frame.frame_id)
            for observation in batch.observations:
                if observation.observation_id in observation_ids:
                    raise ValueError(
                        "observation IDs must be unique across the fixture"
                    )
                observation_ids.add(observation.observation_id)
            if (
                previous_sequence_number is not None
                and frame.sequence_number <= previous_sequence_number
            ):
                raise ValueError("frame sequence numbers must be strictly increasing")
            if previous_timestamp is not None and frame.captured_at < previous_timestamp:
                raise ValueError("frame timestamps must be non-decreasing")
            previous_sequence_number = frame.sequence_number
            previous_timestamp = frame.captured_at

    def replay(self) -> Iterator[FrameObservations]:
        """Yield every recorded frame in order, including empty frames."""
        yield from self.frames

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fixture_id": self.fixture_id,
            "source_id": self.source_id,
            "sensor_id": self.sensor_id,
            "frames": [batch.to_dict() for batch in self.frames],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObservationSequence:
        return cls(
            fixture_id=payload["fixture_id"],
            source_id=payload["source_id"],
            sensor_id=payload["sensor_id"],
            frames=tuple(
                FrameObservations.from_dict(item) for item in payload["frames"]
            ),
            schema_version=payload["schema_version"],
        )


def load_observation_sequence(path: str | Path) -> ObservationSequence:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observation fixture root must be a JSON object")
    return ObservationSequence.from_dict(payload)


def save_observation_sequence(
    sequence: ObservationSequence,
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(sequence.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

