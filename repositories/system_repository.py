from __future__ import annotations

from typing import Dict

from myutils.query import get_connection, init_db


class SystemRepository:
    def overview_counts(self) -> Dict[str, int]:
        init_db()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM movies),
                    (SELECT COUNT(*) FROM user),
                    (SELECT COUNT(*) FROM comments),
                    (SELECT COUNT(*) FROM rl_experience WHERE status = 'pending'),
                    (SELECT COUNT(*) FROM model_registry),
                    (SELECT COUNT(*) FROM experiment_runs)
                """
            )
            row = cursor.fetchone() or (0, 0, 0, 0, 0, 0)
            return {
                "movie_count": int(row[0] or 0),
                "user_count": int(row[1] or 0),
                "comment_count": int(row[2] or 0),
                "pending_experiences": int(row[3] or 0),
                "model_version_count": int(row[4] or 0),
                "experiment_run_count": int(row[5] or 0),
            }
        finally:
            conn.close()
