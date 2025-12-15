from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

from myutils.query import get_connection, init_db


class BehaviorRepository:
    def movie_exists(self, movie_id: int) -> bool:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM movies WHERE id = ?", (movie_id,))
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def upsert_behavior(
        self,
        user_email: str,
        movie_id: int,
        behavior_type: int,
        score: Optional[float],
        create_time: str,
    ) -> bool:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM user_behavior
                WHERE user_email = ? AND movie_id = ? AND behavior_type = ?
                """,
                (user_email, movie_id, behavior_type),
            )
            exists = cursor.fetchone()
            if exists and behavior_type == 1:
                cursor.execute(
                    """
                    UPDATE user_behavior
                    SET score = ?, create_time = ?
                    WHERE user_email = ? AND movie_id = ? AND behavior_type = ?
                    """,
                    (score, create_time, user_email, movie_id, behavior_type),
                )
                conn.commit()
                return False
            elif not exists:
                cursor.execute(
                    """
                    INSERT INTO user_behavior(user_email, movie_id, behavior_type, score, create_time)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (user_email, movie_id, behavior_type, score, create_time),
                )
                conn.commit()
                return True
            conn.commit()
            return False
        finally:
            conn.close()

    def list_user_behaviors(self, user_email: str) -> List[Dict]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ub.movie_id, ub.behavior_type, ub.score, m.title, ub.create_time
                FROM user_behavior ub
                LEFT JOIN movies m ON m.id = ub.movie_id
                WHERE ub.user_email = ?
                ORDER BY ub.create_time DESC, ub.id DESC
                """,
                (user_email,),
            )
            return [
                {
                    "movie_id": row[0],
                    "type": row[1],
                    "score": row[2],
                    "title": row[3],
                    "create_time": row[4],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def list_rating_events(self) -> List[Dict]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_email, movie_id, score, create_time
                FROM user_behavior
                WHERE behavior_type = 1 AND score IS NOT NULL
                ORDER BY user_email ASC, create_time ASC, id ASC
                """
            )
            return [
                {
                    "user_email": str(row[0]),
                    "movie_id": int(row[1]),
                    "score": float(row[2] or 0.0),
                    "create_time": row[3],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def list_interaction_events(self) -> List[Dict]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_email, movie_id, behavior_type, score, create_time
                FROM user_behavior
                ORDER BY user_email ASC, create_time ASC, id ASC
                """
            )
            return [
                {
                    "user_email": str(row[0]),
                    "movie_id": int(row[1]),
                    "behavior_type": int(row[2]),
                    "score": float(row[3]) if row[3] is not None else None,
                    "create_time": row[4],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def list_user_history_records(self, user_email: str) -> List[Dict]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    ub.movie_id,
                    ub.behavior_type,
                    ub.score,
                    m.title,
                    m.types,
                    m.directors,
                    m.country,
                    m.rate,
                    m.comment_len,
                    m.release_year,
                    m.duration,
                    m.summary
                FROM user_behavior ub
                JOIN movies m ON m.id = ub.movie_id
                WHERE ub.user_email = ?
                ORDER BY ub.create_time DESC, ub.id DESC
                """,
                (user_email,),
            )
            rows = cursor.fetchall()
            history = []
            for row in rows:
                history.append(
                    {
                        "movie_id": int(row[0]),
                        "behavior_type": int(row[1]),
                        "score": float(row[2]) if row[2] is not None else None,
                        "title": row[3],
                        "types_list": [item for item in (row[4] or "").split("/") if item],
                        "directors_list": [item for item in (row[5] or "").split("/") if item],
                        "country_list": [item for item in (row[6] or "").split("/") if item],
                        "rate": float(row[7] or 0),
                        "comment_len": int(row[8] or 0),
                        "release_year": int(row[9] or 0),
                        "duration": int(row[10] or 0),
                        "summary": row[11] or "",
                    }
                )
            return history
        finally:
            conn.close()

    def behavior_summary_counts(self) -> Dict[str, int]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN behavior_type = 1 AND score IS NOT NULL THEN 1 ELSE 0 END)
                FROM user_behavior
                """
            )
            row = cursor.fetchone() or (0, 0)
            return {
                "behavior_count": int(row[0] or 0),
                "rating_count": int(row[1] or 0),
            }
        finally:
            conn.close()

    def get_behavior_snapshot(self, user_email: str, movie_id: int) -> Dict:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT behavior_type, score
                FROM user_behavior
                WHERE user_email = ? AND movie_id = ?
                """,
                (user_email, movie_id),
            )
            snapshot = {"rated": False, "score": None, "collected": False, "watched": False}
            for behavior_type, score in cursor.fetchall():
                if behavior_type == 1:
                    snapshot["rated"] = True
                    snapshot["score"] = score
                elif behavior_type == 2:
                    snapshot["collected"] = True
                elif behavior_type == 3:
                    snapshot["watched"] = True
            return snapshot
        finally:
            conn.close()

    def list_feedback_rows(self) -> List[Tuple]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_email, movie_id, behavior_type, score
                FROM user_behavior
                ORDER BY create_time ASC, id ASC
                """
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def get_movie_exposure_map(self) -> Dict[int, int]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT movie_id, COUNT(*) AS total
                FROM user_behavior
                GROUP BY movie_id
                """
            )
            return {int(movie_id): int(total) for movie_id, total in cursor.fetchall()}
        finally:
            conn.close()

    def insert_experience(
        self,
        user_email: str,
        movie_id: int,
        behavior_type: int,
        reward: float,
        old_prob: Optional[float],
        payload: Dict,
        status: str = "pending",
    ) -> None:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO rl_experience(user_email, movie_id, behavior_type, reward, old_prob, payload_json, status)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (user_email, movie_id, behavior_type, reward, old_prob, json.dumps(payload, ensure_ascii=False), status),
            )
            conn.commit()
        finally:
            conn.close()

    def pending_experience_count(self) -> int:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rl_experience WHERE status = 'pending'")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def list_pending_experiences(self, limit: int) -> List[Tuple]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_email, movie_id, reward, old_prob, payload_json
                FROM rl_experience
                WHERE status = 'pending'
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def mark_experiences_processed(self, ids: Sequence[int], model_version: str) -> None:
        if not ids:
            return
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            placeholders = ", ".join(["?"] * len(ids))
            cursor.execute(
                f"UPDATE rl_experience SET status = 'processed', model_version = ? WHERE id IN ({placeholders})",
                (model_version, *ids),
            )
            conn.commit()
        finally:
            conn.close()
