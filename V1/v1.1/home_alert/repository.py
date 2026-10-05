"""SQLite persistence for person events."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .events import Event, EventState, IdentityKind


SCHEMA_VERSION = 1


class EventRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise RuntimeError(f"Unsupported database schema version: {version}")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    identity_kind TEXT NOT NULL CHECK(identity_kind IN ('known','unknown','uncertain')),
                    state TEXT NOT NULL CHECK(state IN ('recording','complete','failed')),
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    person_name TEXT,
                    confidence REAL,
                    track_ids TEXT NOT NULL DEFAULT '[]',
                    snapshot_path TEXT,
                    video_path TEXT,
                    protected INTEGER NOT NULL DEFAULT 0 CHECK(protected IN (0,1)),
                    reviewed INTEGER NOT NULL DEFAULT 0 CHECK(reviewed IN (0,1)),
                    alert_sent INTEGER NOT NULL DEFAULT 0 CHECK(alert_sent IN (0,1)),
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS events_started_at_idx ON events(started_at DESC)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS events_retention_idx ON events(protected, identity_kind, ended_at)"
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def add(self, event: Event) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO events (
                    id, identity_kind, state, started_at, ended_at, person_name,
                    confidence, track_ids, snapshot_path, video_path, protected,
                    reviewed, alert_sent, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._values(event),
            )

    def save(self, event: Event) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE events SET identity_kind=?, state=?, started_at=?, ended_at=?,
                    person_name=?, confidence=?, track_ids=?, snapshot_path=?,
                    video_path=?, protected=?, reviewed=?, alert_sent=?, created_at=?
                    WHERE id=?""",
                self._values(event)[1:] + (event.id,),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown event: {event.id}")

    def get(self, event_id: str) -> Event | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._from_row(row) if row else None

    def list(self, *, limit: int = 100, include_recording: bool = True) -> list[Event]:
        if limit < 1:
            raise ValueError("Limit must be positive")
        where = "" if include_recording else "WHERE state != 'recording'"
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM events {where} ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def retention_candidates(self) -> list[Event]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM events WHERE protected = 0 AND state != 'recording'
                   ORDER BY COALESCE(ended_at, started_at) ASC"""
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, event_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM events WHERE id = ?", (event_id,))
        return cursor.rowcount == 1

    @staticmethod
    def _values(event: Event) -> tuple[object, ...]:
        return (
            event.id, event.identity_kind.value, event.state.value,
            event.started_at.isoformat(), event.ended_at.isoformat() if event.ended_at else None,
            event.person_name, event.confidence, json.dumps(event.track_ids),
            event.snapshot_path, event.video_path, int(event.protected), int(event.reviewed),
            int(event.alert_sent), event.created_at.isoformat(),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"], identity_kind=IdentityKind(row["identity_kind"]),
            state=EventState(row["state"]), started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            person_name=row["person_name"], confidence=row["confidence"],
            track_ids=tuple(json.loads(row["track_ids"])), snapshot_path=row["snapshot_path"],
            video_path=row["video_path"], protected=bool(row["protected"]),
            reviewed=bool(row["reviewed"]), alert_sent=bool(row["alert_sent"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
