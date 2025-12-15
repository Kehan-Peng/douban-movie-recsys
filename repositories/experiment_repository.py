from __future__ import annotations

import json
from typing import Dict, List, Optional

from myutils.query import get_connection, init_db


class ExperimentRepository:
    def save_run(self, run_name: str, metrics_payload: Dict, note: str = "") -> Dict:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiment_runs(run_name, metrics_json, note)
                VALUES(?, ?, ?)
                """,
                (run_name, json.dumps(metrics_payload, ensure_ascii=False), note),
            )
            conn.commit()
            cursor.execute(
                """
                SELECT id, run_name, metrics_json, note, created_at
                FROM experiment_runs
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            return self._row_to_dict(row)
        finally:
            conn.close()

    def list_runs(self, limit: int = 20) -> List[Dict]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, run_name, metrics_json, note, created_at
                FROM experiment_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def latest_run(self) -> Optional[Dict]:
        rows = self.list_runs(limit=1)
        return rows[0] if rows else None

    @staticmethod
    def _row_to_dict(row) -> Dict:
        run_id, run_name, metrics_json, note, created_at = row
        return {
            "id": int(run_id),
            "run_name": run_name,
            "metrics": json.loads(metrics_json or "{}"),
            "note": note,
            "created_at": created_at,
        }
