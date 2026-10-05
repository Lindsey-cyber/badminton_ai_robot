"""Small SQLite event log for one-process sessions.

Event ID makes exact replays idempotent. A different event with the same
session sequence is a conflict, not an update or a silent overwrite.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .events import Event, EventType


class EventStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                sequence INTEGER NOT NULL CHECK(sequence >= 0),
                timestamp_utc TEXT NOT NULL,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(session_id, sequence)
            );
        """)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def create_session(self, session_id: str, started_at: datetime | None = None) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        started_at = started_at or datetime.now(timezone.utc)
        if started_at.utcoffset() is None:
            raise ValueError("Session timestamp must include a timezone")
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO sessions(session_id, started_at_utc) VALUES (?, ?)",
                (session_id, started_at.astimezone(timezone.utc).isoformat()),
            )

    def append(self, event: Event) -> bool:
        """Return False for an identical replay; raise for conflicting IDs/order."""
        if event.timestamp_utc.utcoffset() is None:
            raise ValueError("Event timestamp must include a timezone")
        record = (event.event_id, event.session_id, event.sequence,
                  event.timestamp_utc.astimezone(timezone.utc).isoformat(),
                  event.type.value, json.dumps(dict(event.payload), sort_keys=True, allow_nan=False))
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO events(event_id, session_id, sequence, timestamp_utc, type, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)", record,
                )
            return True
        except sqlite3.IntegrityError as exc:
            existing = self._connection.execute(
                "SELECT event_id, session_id, sequence, timestamp_utc, type, payload_json "
                "FROM events WHERE event_id = ?", (event.event_id,),
            ).fetchone()
            if existing == record:
                return False
            raise ValueError("Event ID, session sequence or session reference conflicts") from exc

    def next_sequence(self, session_id: str) -> int:
        row = self._connection.execute(
            "SELECT MAX(sequence) FROM events WHERE session_id = ?", (session_id,),
        ).fetchone()
        return (row[0] + 1) if row[0] is not None else 0

    def read_events(self, session_id: str) -> list[Event]:
        rows = self._connection.execute(
            "SELECT event_id, session_id, sequence, timestamp_utc, type, payload_json "
            "FROM events WHERE session_id = ? ORDER BY sequence", (session_id,),
        ).fetchall()
        return [Event(event_id, stored_session, sequence,
                      datetime.fromisoformat(timestamp), EventType(event_type), json.loads(payload))
                for event_id, stored_session, sequence, timestamp, event_type, payload in rows]
