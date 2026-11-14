"""Ordered, replayable session events. Render Free storage is explicitly ephemeral."""
import json
import os
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


class EventLog:
    def __init__(self, path=None):
        self.connection = sqlite3.connect(path or Path(tempfile.gettempdir()) / f"mars-pathfinder-{os.getpid()}.sqlite", check_same_thread=False)
        self.lock = threading.Lock()
        self.connection.execute("CREATE TABLE IF NOT EXISTS events (sequence_id INTEGER PRIMARY KEY AUTOINCREMENT, mission_id TEXT, payload TEXT)")

    def append(self, mission_id: str, event_type: str, message: str, **details):
        event = {"mission_id":mission_id, "type":event_type, "message":message,
                 "timestamp":datetime.now(timezone.utc).isoformat(), **details}
        with self.lock:
            cursor = self.connection.execute("INSERT INTO events(mission_id,payload) VALUES (?,?)",(mission_id,json.dumps(event)))
            self.connection.commit()
            event["sequence_id"] = cursor.lastrowid
        return event

    def read(self, mission_id: str, after: int = 0):
        with self.lock:
            records = self.connection.execute("SELECT sequence_id,payload FROM events WHERE mission_id=? AND sequence_id>? ORDER BY sequence_id",(mission_id,after)).fetchall()
        return [{**json.loads(payload),"sequence_id":seq} for seq,payload in records]
