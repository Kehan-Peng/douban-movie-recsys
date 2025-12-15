from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from repositories.behavior_repository import BehaviorRepository


class BehaviorService:
    VALID_BEHAVIOR_TYPES = {1, 2, 3}

    def __init__(self) -> None:
        self.repository = BehaviorRepository()

    def validate(self, movie_id: int, behavior_type: int, score: Optional[float]) -> None:
        if behavior_type not in self.VALID_BEHAVIOR_TYPES:
            raise ValueError("不支持的行为类型。")
        if not self.repository.movie_exists(movie_id):
            raise ValueError("电影不存在，无法提交行为。")
        if behavior_type == 1:
            if score is None:
                raise ValueError("评分行为必须提供 score。")
            if score < 0 or score > 10:
                raise ValueError("评分必须在 0-10 分之间。")

    def add_behavior(
        self,
        user_email: str,
        movie_id: int,
        behavior_type: int,
        score: Optional[float],
        create_time: Optional[str] = None,
    ) -> str:
        self.save_behavior(user_email, movie_id, behavior_type, score, create_time=create_time)
        return "success"

    def save_behavior(
        self,
        user_email: str,
        movie_id: int,
        behavior_type: int,
        score: Optional[float],
        create_time: Optional[str] = None,
    ) -> bool:
        self.validate(movie_id, behavior_type, score)
        resolved_create_time = create_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.repository.upsert_behavior(user_email, movie_id, behavior_type, score, resolved_create_time)

    def get_user_behavior(self, user_email: str) -> List[Dict]:
        return self.repository.list_user_behaviors(user_email)

    def get_rating_events(self) -> List[Dict]:
        return self.repository.list_rating_events()

    def get_interaction_events(self) -> List[Dict]:
        return self.repository.list_interaction_events()

    def get_user_history_records(self, user_email: str) -> List[Dict]:
        return self.repository.list_user_history_records(user_email)

    def get_behavior_summary_counts(self) -> Dict[str, int]:
        return self.repository.behavior_summary_counts()

    def get_behavior_snapshot(self, user_email: Optional[str], movie_id: int) -> Dict:
        if not user_email:
            return {}
        return self.repository.get_behavior_snapshot(user_email, movie_id)
