from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from home_alert.events import Event, EventState, IdentityKind
from home_alert.repository import EventRepository


class EventRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = EventRepository(self.root / "events.sqlite3")
        self.now = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_event_lifecycle_round_trip(self) -> None:
        event = Event(identity_kind=IdentityKind.UNKNOWN, started_at=self.now, track_ids=(4,))
        self.repository.add(event)
        event.state = EventState.COMPLETE
        event.ended_at = self.now + timedelta(seconds=12)
        event.video_path = f"{event.id}/clip.mp4"
        event.alert_sent = True
        self.repository.save(event)

        stored = self.repository.get(event.id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.state, EventState.COMPLETE)
        self.assertEqual(stored.track_ids, (4,))
        self.assertTrue(stored.alert_sent)

    def test_known_event_requires_a_name(self) -> None:
        with self.assertRaises(ValueError):
            Event(identity_kind=IdentityKind.KNOWN, started_at=self.now)

    def test_protection_and_review_state_are_persisted(self) -> None:
        event = Event(identity_kind=IdentityKind.UNKNOWN, started_at=self.now)
        self.repository.add(event)
        event.protected = True
        event.reviewed = True
        self.repository.save(event)
        stored = self.repository.get(event.id)
        self.assertTrue(stored.protected)
        self.assertTrue(stored.reviewed)


if __name__ == "__main__":
    unittest.main()
