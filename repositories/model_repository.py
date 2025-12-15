from __future__ import annotations

import json
from typing import Dict, List, Optional

from myutils.query import get_connection, init_db


class ModelRepository:
    def active_model(self, model_name: str) -> Optional[Dict]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT version_tag, storage_path, metrics_json, note, created_at
                FROM model_registry
                WHERE model_name = ? AND is_active = 1
                ORDER BY id DESC
                LIMIT 1
                """,
                (model_name,),
            )
            row = cursor.fetchone()
            return self._row_to_dict(row, active=True) if row else None
        finally:
            conn.close()

    def list_versions(self, model_name: str) -> List[Dict]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT version_tag, storage_path, metrics_json, note, is_active, created_at
                FROM model_registry
                WHERE model_name = ?
                ORDER BY id DESC
                """,
                (model_name,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "version_tag": row[0],
                    "storage_path": row[1],
                    "metrics": json.loads(row[2] or "{}"),
                    "note": row[3],
                    "is_active": bool(row[4]),
                    "created_at": row[5],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def version_exists(self, model_name: str, version_tag: str) -> bool:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM model_registry WHERE model_name = ? AND version_tag = ?",
                (model_name, version_tag),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def activate_version(self, model_name: str, version_tag: str) -> None:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE model_registry SET is_active = 0 WHERE model_name = ?", (model_name,))
            cursor.execute(
                "UPDATE model_registry SET is_active = 1 WHERE model_name = ? AND version_tag = ?",
                (model_name, version_tag),
            )
            conn.commit()
        finally:
            conn.close()

    def save_version(self, model_name: str, version_tag: str, storage_path: str, metrics: Dict, note: str) -> Dict:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE model_registry SET is_active = 0 WHERE model_name = ?", (model_name,))
            cursor.execute(
                """
                INSERT INTO model_registry(model_name, version_tag, storage_path, metrics_json, note, is_active)
                VALUES(?, ?, ?, ?, ?, 1)
                """,
                (model_name, version_tag, storage_path, json.dumps(metrics, ensure_ascii=False), note),
            )
            conn.commit()
            return {
                "version_tag": version_tag,
                "storage_path": storage_path,
                "metrics": metrics,
                "note": note,
            }
        finally:
            conn.close()

    def delete_version(self, model_name: str, version_tag: str) -> None:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM model_registry WHERE model_name = ? AND version_tag = ?",
                (model_name, version_tag),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_dict(row, active: bool = False) -> Dict:
        version_tag, storage_path, metrics_json, note, created_at = row
        return {
            "version_tag": version_tag,
            "storage_path": storage_path,
            "metrics": json.loads(metrics_json or "{}"),
            "note": note,
            "is_active": active,
            "created_at": created_at,
        }
