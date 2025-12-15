from __future__ import annotations

from pathlib import Path
from typing import Dict

from repositories.system_repository import SystemRepository
from services.behavior_service import BehaviorService


class SystemService:
    def __init__(self) -> None:
        self.repository = SystemRepository()
        self.behavior_service = BehaviorService()

    def overview_counts(self, db_path: str, movie_feature_cache_size: int) -> Dict:
        system_counts = self.repository.overview_counts()
        behavior_counts = self.behavior_service.get_behavior_summary_counts()
        movie_count = system_counts["movie_count"]
        user_count = system_counts["user_count"]
        rating_count = behavior_counts["rating_count"]
        density = rating_count / max(movie_count * user_count, 1)
        db_size_kb = round(Path(db_path).stat().st_size / 1024, 2) if Path(db_path).exists() else 0
        return {
            **system_counts,
            **behavior_counts,
            "interaction_density": round(density, 4),
            "db_size_kb": db_size_kb,
            "movie_feature_cache_size": movie_feature_cache_size,
        }
