import os
import tempfile
import pytest
from app.core.storage import AlertDatabase


def test_sqlite_alert_storage_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_alerts.db")
        db = AlertDatabase(db_path=db_path)

        assert os.path.exists(db_path)
        assert len(db.get_alerts()) == 0

        # Save an alert
        test_payload = {
            "event_id": "evt_test_001",
            "timestamp": "2026-10-25T12:00:00.000Z",
            "alert_type": "PEDESTRIAN_HAZARD",
            "confidence": 0.94,
            "zone": "A-2",
            "details": {"test": True}
        }
        success = db.save_alert(test_payload)
        assert success is True

        # Retrieve
        alerts = db.get_alerts(limit=10)
        assert len(alerts) == 1
        assert alerts[0]["event_id"] == "evt_test_001"
        assert alerts[0]["confidence"] == 0.94

        # Purge
        purged = db.purge_old_alerts(max_age_days=0)
        assert purged == 1
        assert len(db.get_alerts()) == 0
