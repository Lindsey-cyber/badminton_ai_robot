"""SQLite replay, ordering and conflict handling without external services."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.events import EventDispatcher, EventType, SessionEventEmitter
from badminton_ai.storage import EventStore


class StorageTests(unittest.TestCase):
    def test_replay_and_restart_keep_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            with EventStore(path) as store:
                store.create_session("s")
                dispatcher = EventDispatcher()
                dispatcher.subscribe(EventType.SHOT_CANDIDATE, store.append)
                emitter = SessionEventEmitter("s", dispatcher)
                first = emitter.emit(EventType.SHOT_CANDIDATE, {"frame_index": 2})
                self.assertFalse(store.append(first))
                self.assertEqual(store.next_sequence("s"), 1)

            with EventStore(path) as reopened:
                dispatcher = EventDispatcher()
                dispatcher.subscribe(EventType.SHOT_CANDIDATE, reopened.append)
                emitter = SessionEventEmitter("s", dispatcher, reopened.next_sequence("s"))
                second = emitter.emit(EventType.SHOT_CANDIDATE, {"frame_index": 8})
                self.assertEqual([event.sequence for event in reopened.read_events("s")], [0, 1])
                self.assertEqual(second.sequence, 1)
                self.assertEqual([e.sequence for e in reopened.read_events("s", 0, 1)], [1])
                self.assertEqual(reopened.read_events("s", 1), [])
                self.assertTrue(reopened.session_exists("s"))
                self.assertEqual([s["session_id"] for s in reopened.list_sessions()], ["s"])
                with self.assertRaises(ValueError):
                    reopened.append(replace(second, event_id="another-id"))
                with self.assertRaises(ValueError):
                    reopened.append(replace(first, payload={"frame_index": 99}))

    def test_unknown_session_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with EventStore(Path(directory) / "events.sqlite3") as store:
                emitter = SessionEventEmitter("missing", EventDispatcher())
                event = emitter.emit(EventType.SHOT_CANDIDATE, {"frame_index": 1})
                with self.assertRaises(ValueError):
                    store.append(event)


if __name__ == "__main__":
    unittest.main()
