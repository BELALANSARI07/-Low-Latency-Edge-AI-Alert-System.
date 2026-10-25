"""
Persistent SQLite Alert Storage Layer.
Provides lightweight, zero-dependency persistence for edge event logs,
audit trails, and offline alert buffering with automatic retention management.
"""

import os
import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class AlertDatabase:
    """SQLite persistence manager for edge alert history and telemetry audits."""

    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates connection with row factory configured."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes database schema and indexes."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    zone TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_event_id ON alerts (event_id)
            """)
            conn.commit()
        logger.info(f"Initialized SQLite alert database at {self.db_path}.")

    def save_alert(self, payload: Dict[str, Any]) -> bool:
        """Stores a dispatched alert record."""
        event_id = payload.get("event_id", f"evt_{int(datetime.now(timezone.utc).timestamp() * 1000)}")
        timestamp = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
        alert_type = payload.get("alert_type", "UNKNOWN")
        confidence = float(payload.get("confidence", 0.0))
        zone = payload.get("zone", "DEFAULT")
        payload_json = json.dumps(payload)

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO alerts (event_id, timestamp, alert_type, confidence, zone, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, timestamp, alert_type, confidence, zone, payload_json)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to persist alert event {event_id}: {e}")
            return False

    def get_alerts(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieves paginated alerts sorted from most recent to oldest."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT payload_json FROM alerts
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                )
                rows = cursor.fetchall()
                return [json.loads(row["payload_json"]) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve alerts: {e}")
            return []

    def purge_old_alerts(self, max_age_days: int = 7) -> int:
        """Purges alerts older than max_age_days to bound disk consumption."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM alerts WHERE timestamp < ?",
                    (cutoff,)
                )
                purged_count = cursor.rowcount
                conn.commit()
            logger.info(f"Purged {purged_count} alerts older than {max_age_days} days.")
            return purged_count
        except Exception as e:
            logger.error(f"Failed to purge old alerts: {e}")
            return 0
