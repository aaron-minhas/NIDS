"""
SQLite-backed alert log -- audit trail for forensic review.
"""
from __future__ import annotations
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

_DDL = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    flow_index INTEGER,
    pred_class TEXT NOT NULL,
    confidence REAL,
    rule_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    because TEXT,
    actions_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_class ON alerts(pred_class);
"""


class AlertLog:
    def __init__(self, db_path="reports/alerts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(_DDL)
            conn.commit()

    def log_predictions(self, predictions, skip_informational=True):
        # Assign each row a unique microsecond timestamp by base + offset.
        # Without this, an entire 50K-row batch ends up with the SAME ts_utc,
        # making the audit trail look like junk in the UI.
        rows = []
        base = datetime.now(timezone.utc)
        offset_us = 0
        from datetime import timedelta as _td
        for p in predictions:
            for adv in p.advisories:
                if skip_informational and adv.severity == "Informational":
                    continue
                ts = (base + _td(microseconds=offset_us)).isoformat(timespec="microseconds")
                rows.append((ts, p.flow_index, p.predicted_class, p.confidence,
                             adv.rule_id, adv.severity, adv.title, adv.because,
                             json.dumps(adv.recommended_actions)))
                offset_us += 1
        if not rows:
            return 0
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executemany(
                "INSERT INTO alerts (ts_utc, flow_index, pred_class, confidence, rule_id, severity, title, because, actions_json) VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        return len(rows)

    def recent(self, limit=100, since=None):
        """Return recent alerts. If `since` (ISO ts string) provided, only alerts AFTER that time."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            if since:
                return pd.read_sql_query(
                    "SELECT * FROM alerts WHERE ts_utc > ? ORDER BY id DESC LIMIT ?",
                    conn, params=(since, limit),
                )
            return pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", conn, params=(limit,))

    def severity_breakdown(self, since=None):
        """Severity histogram. If `since` provided, only alerts AFTER that time."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            if since:
                return pd.read_sql_query(
                    "SELECT severity, COUNT(*) AS n FROM alerts WHERE ts_utc > ? GROUP BY severity ORDER BY n DESC",
                    conn, params=(since,),
                )
            return pd.read_sql_query("SELECT severity, COUNT(*) AS n FROM alerts GROUP BY severity ORDER BY n DESC", conn)

    def clear(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM alerts")
            conn.commit()
