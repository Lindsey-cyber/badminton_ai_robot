"""Small local API over the SQLite event log.

WebSocket clients catch up by sequence and poll the durable log for new events.
This works when the video producer and API run in separate processes.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .events import Event, EventType
from .storage import EventStore


class SessionInput(BaseModel):
    session_id: str = Field(min_length=1)


class EventInput(BaseModel):
    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    timestamp_utc: datetime
    type: EventType
    payload: dict[str, str | int | float]


def _event_json(event: Event) -> dict:
    return {"event_id": event.event_id, "session_id": event.session_id,
            "sequence": event.sequence, "timestamp_utc": event.timestamp_utc.isoformat(),
            "type": event.type.value, "payload": dict(event.payload)}


def create_app(db_path: Path) -> FastAPI:
    app = FastAPI(title="Badminton training event API")

    @app.get("/health")
    def health() -> dict[str, str]:
        with EventStore(db_path) as store:
            store.list_sessions()
        return {"status": "ok"}

    @app.post("/sessions", status_code=201)
    def create_session(body: SessionInput) -> dict[str, str]:
        with EventStore(db_path) as store:
            store.create_session(body.session_id)
        return {"session_id": body.session_id}

    @app.get("/sessions")
    def list_sessions() -> list[dict[str, str]]:
        with EventStore(db_path) as store:
            return store.list_sessions()

    @app.post("/events")
    def append_event(body: EventInput) -> dict[str, bool]:
        event = Event(body.event_id, body.session_id, body.sequence,
                      body.timestamp_utc, body.type, body.payload)
        try:
            with EventStore(db_path) as store:
                inserted = store.append(event)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"inserted": inserted}

    @app.get("/sessions/{session_id}/events")
    def read_events(session_id: str, after_sequence: int = Query(-1, ge=-1),
                    limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
        with EventStore(db_path) as store:
            if not store.session_exists(session_id):
                raise HTTPException(status_code=404, detail="Unknown session")
            return [_event_json(event) for event in store.read_events(
                session_id, after_sequence, limit)]

    @app.websocket("/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str,
                     after_sequence: int = -1) -> None:
        await websocket.accept()
        with EventStore(db_path) as store:
            if after_sequence < -1 or not store.session_exists(session_id):
                await websocket.close(code=4404)
                return
        cursor = after_sequence
        try:
            while True:
                with EventStore(db_path) as store:
                    events = store.read_events(session_id, cursor, limit=100)
                for event in events:
                    await websocket.send_json(_event_json(event))
                    cursor = event.sequence
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.25)
                except TimeoutError:
                    pass
        except WebSocketDisconnect:
            # A disconnected client does not affect the writer or other readers.
            return

    return app
