"""API replay, duplicate handling and cross-connection WebSocket delivery."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
    from badminton_ai.api import create_app
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Install requirements-api.txt to run API tests")
class ApiTests(unittest.TestCase):
    def test_session_event_replay_and_websocket(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory) / "events.sqlite3")) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ok"})
                self.assertEqual(client.get("/sessions/missing/events").status_code, 404)
                self.assertEqual(client.post("/sessions", json={"session_id": "s"}).status_code, 201)
                self.assertEqual(len(client.get("/sessions").json()), 1)
                body = {"event_id": "e0", "session_id": "s", "sequence": 0,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "type": "ShotCandidate", "payload": {"frame_index": 3}}
                self.assertEqual(client.post("/events", json=body).json(), {"inserted": True})
                self.assertEqual(client.post("/events", json=body).json(), {"inserted": False})
                self.assertEqual(client.post("/events", json={**body, "event_id": "conflict"}).status_code, 409)
                with client.websocket_connect("/sessions/s/stream?after_sequence=-1") as websocket:
                    self.assertEqual(websocket.receive_json()["event_id"], "e0")
                    second = {**body, "event_id": "e1", "sequence": 1}
                    self.assertEqual(client.post("/events", json=second).status_code, 200)
                    self.assertEqual(websocket.receive_json()["event_id"], "e1")
                    metric = {**body, "event_id": "m2", "sequence": 2,
                              "type": "PerformanceMetric", "payload": {"processed_fps": 18.0}}
                    self.assertEqual(client.post("/events", json=metric).status_code, 200)
                    self.assertEqual(websocket.receive_json()["type"], "PerformanceMetric")
                self.assertEqual([e["sequence"] for e in client.get(
                    "/sessions/s/events?after_sequence=0&limit=1").json()], [1])
                self.assertEqual(client.get("/sessions/s/events?limit=0").status_code, 422)


if __name__ == "__main__":
    unittest.main()
