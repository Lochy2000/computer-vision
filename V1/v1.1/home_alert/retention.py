"""Retention cleanup for event metadata and local media."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .config import RetentionConfig
from .events import Event, IdentityKind
from .repository import EventRepository


@dataclass(frozen=True)
class CleanupResult:
    deleted_event_ids: tuple[str, ...]
    bytes_removed: int


class RetentionService:
    def __init__(self, repository: EventRepository, media_root: Path, config: RetentionConfig) -> None:
        self.repository = repository
        self.media_root = media_root.resolve()
        self.config = config

    def cleanup(self, now: datetime) -> CleanupResult:
        if now.tzinfo is None:
            raise ValueError("Cleanup time must be timezone-aware")
        candidates = self.repository.retention_candidates()
        expired = {event.id for event in candidates if self._expired(event, now)}
        remaining = [event for event in candidates if event.id not in expired]
        total_bytes = self._total_media_bytes() - sum(
            self._media_size(event) for event in candidates if event.id in expired
        )
        over_budget: set[str] = set()
        for event in remaining:
            if total_bytes <= self.config.maximum_media_bytes:
                break
            size = self._media_size(event)
            over_budget.add(event.id)
            total_bytes -= size

        removed_bytes = 0
        deleted: list[str] = []
        for event in candidates:
            if event.id not in expired | over_budget:
                continue
            removed_bytes += self._delete_media(event)
            if self.repository.delete(event.id):
                deleted.append(event.id)
        return CleanupResult(tuple(deleted), removed_bytes)

    def _expired(self, event: Event, now: datetime) -> bool:
        days = {
            IdentityKind.KNOWN: self.config.known_days,
            IdentityKind.UNKNOWN: self.config.unknown_days,
            IdentityKind.UNCERTAIN: self.config.uncertain_days,
        }[event.identity_kind]
        return (event.ended_at or event.started_at) < now - timedelta(days=days)

    def _event_paths(self, event: Event) -> tuple[Path, ...]:
        paths: list[Path] = []
        for value in (event.snapshot_path, event.video_path):
            if not value:
                continue
            path = (self.media_root / value).resolve()
            if self.media_root not in path.parents:
                raise ValueError(f"Event media path escapes the media directory: {value}")
            paths.append(path)
        return tuple(paths)

    def _media_size(self, event: Event) -> int:
        return sum(path.stat().st_size for path in self._event_paths(event) if path.is_file())

    def _total_media_bytes(self) -> int:
        if not self.media_root.exists():
            return 0
        return sum(path.stat().st_size for path in self.media_root.rglob("*") if path.is_file())

    def _delete_media(self, event: Event) -> int:
        removed = 0
        parents: set[Path] = set()
        for path in self._event_paths(event):
            if path.is_file():
                removed += path.stat().st_size
                path.unlink()
                parents.add(path.parent)
        for parent in sorted(parents, key=lambda item: len(item.parts), reverse=True):
            if parent != self.media_root and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        return removed
