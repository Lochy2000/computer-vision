"""Domain types for recorded person events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4


class IdentityKind(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    UNCERTAIN = "uncertain"


class EventState(StrEnum):
    RECORDING = "recording"
    COMPLETE = "complete"
    FAILED = "failed"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Event:
    identity_kind: IdentityKind
    started_at: datetime
    id: str = field(default_factory=lambda: uuid4().hex)
    state: EventState = EventState.RECORDING
    ended_at: datetime | None = None
    person_name: str | None = None
    confidence: float | None = None
    track_ids: tuple[int, ...] = ()
    snapshot_path: str | None = None
    video_path: str | None = None
    protected: bool = False
    reviewed: bool = False
    alert_sent: bool = False
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for value in (self.started_at, self.ended_at, self.created_at):
            if value is not None and value.tzinfo is None:
                raise ValueError("Event datetimes must be timezone-aware")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("Event cannot end before it starts")
        if self.identity_kind is IdentityKind.KNOWN and not self.person_name:
            raise ValueError("Known events require a person name")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
