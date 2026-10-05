from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from home_alert.config import RetentionConfig
from home_alert.events import Event, EventState, IdentityKind
from home_alert.repository import EventRepository
from home_alert.retention import RetentionService


class RetentionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.media = self.root / "media"
        self.media.mkdir()
        self.repository = EventRepository(self.root / "events.sqlite3")
        self.now = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _event(self, *, age_days: int, kind: IdentityKind, size: int, protected: bool = False) -> Event:
        event = Event(
            identity_kind=kind,
            person_name="Loch" if kind is IdentityKind.KNOWN else None,
            state=EventState.COMPLETE,
            started_at=self.now - timedelta(days=age_days, minutes=1),
            ended_at=self.now - timedelta(days=age_days),
            protected=protected,
        )
        directory = self.media / event.id
        directory.mkdir()
        clip = directory / "clip.mp4"
        clip.write_bytes(b"x" * size)
        event.video_path = f"{event.id}/clip.mp4"
        self.repository.add(event)
        return event

    def test_expired_events_are_deleted_but_protected_events_survive(self) -> None:
        expired = self._event(age_days=4, kind=IdentityKind.KNOWN, size=20)
        protected = self._event(age_days=20, kind=IdentityKind.UNKNOWN, size=30, protected=True)
        service = RetentionService(
            self.repository,
            self.media,
            RetentionConfig(known_days=3, unknown_days=14, uncertain_days=14),
        )
        result = service.cleanup(self.now)

        self.assertEqual(result.deleted_event_ids, (expired.id,))
        self.assertEqual(result.bytes_removed, 20)
        self.assertIsNone(self.repository.get(expired.id))
        self.assertIsNotNone(self.repository.get(protected.id))

    def test_storage_budget_removes_oldest_unprotected_event_first(self) -> None:
        oldest = self._event(age_days=2, kind=IdentityKind.UNKNOWN, size=60)
        newest = self._event(age_days=1, kind=IdentityKind.UNKNOWN, size=60)
        service = RetentionService(
            self.repository,
            self.media,
            RetentionConfig(
                known_days=30,
                unknown_days=30,
                uncertain_days=30,
                maximum_media_bytes=70,
            ),
        )
        result = service.cleanup(self.now)

        self.assertEqual(result.deleted_event_ids, (oldest.id,))
        self.assertIsNone(self.repository.get(oldest.id))
        self.assertIsNotNone(self.repository.get(newest.id))

    def test_protected_media_counts_toward_budget_without_being_deleted(self) -> None:
        protected = self._event(
            age_days=3, kind=IdentityKind.UNKNOWN, size=60, protected=True
        )
        removable = self._event(age_days=2, kind=IdentityKind.UNKNOWN, size=50)
        service = RetentionService(
            self.repository,
            self.media,
            RetentionConfig(
                known_days=30,
                unknown_days=30,
                uncertain_days=30,
                maximum_media_bytes=70,
            ),
        )
        result = service.cleanup(self.now)

        self.assertEqual(result.deleted_event_ids, (removable.id,))
        self.assertIsNotNone(self.repository.get(protected.id))
        self.assertIsNone(self.repository.get(removable.id))

    def test_paths_cannot_escape_media_root(self) -> None:
        event = Event(
            identity_kind=IdentityKind.UNKNOWN,
            state=EventState.COMPLETE,
            started_at=self.now - timedelta(days=20),
            ended_at=self.now - timedelta(days=19),
            video_path="../outside.mp4",
        )
        self.repository.add(event)
        service = RetentionService(self.repository, self.media, RetentionConfig())
        with self.assertRaises(ValueError):
            service.cleanup(self.now)


if __name__ == "__main__":
    unittest.main()
