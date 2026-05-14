"""Operator audit trail. Every dashboard action logged for SOC accountability."""
from __future__ import annotations
import sqlite3, json, getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import pandas as pd


class OperatorAudit:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    def _init(self):
        with self._conn() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS operator_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target TEXT,
                details TEXT,
                source_ip TEXT
            )''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_audit_ts ON operator_actions(ts_utc)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_audit_op ON operator_actions(operator_id)')

    def log(self, operator_id: str, action_type: str, target: str = "", details: dict = None, source_ip: str = "127.0.0.1"):
        ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._conn() as c:
            c.execute('INSERT INTO operator_actions(ts_utc, operator_id, action_type, target, details, source_ip) VALUES (?,?,?,?,?,?)',
                     (ts, operator_id, action_type, target,
                      json.dumps(details or {}, default=str), source_ip))

    def recent(self, limit: int = 100, operator_id: str = None) -> pd.DataFrame:
        with self._conn() as c:
            if operator_id:
                df = pd.read_sql_query('SELECT * FROM operator_actions WHERE operator_id=? ORDER BY id DESC LIMIT ?',
                                       c, params=(operator_id, limit))
            else:
                df = pd.read_sql_query('SELECT * FROM operator_actions ORDER BY id DESC LIMIT ?', c, params=(limit,))
        return df

    def summary(self) -> dict:
        with self._conn() as c:
            row = c.execute('SELECT COUNT(*), COUNT(DISTINCT operator_id), MIN(ts_utc), MAX(ts_utc) FROM operator_actions').fetchone()
            actions_by_type = c.execute('SELECT action_type, COUNT(*) FROM operator_actions GROUP BY action_type ORDER BY 2 DESC').fetchall()
        return {
            "total_actions": row[0] if row else 0,
            "unique_operators": row[1] if row else 0,
            "first_action": row[2] if row else None,
            "last_action": row[3] if row else None,
            "by_type": dict(actions_by_type),
        }

    def clear(self):
        with self._conn() as c:
            c.execute('DELETE FROM operator_actions')


def get_default_operator() -> str:
    """Resolve operator id (env override, else system user, else 'analyst')."""
    import os
    return os.environ.get("AIES_OPERATOR_ID") or getpass.getuser() or "analyst"