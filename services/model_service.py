from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from repositories.behavior_repository import BehaviorRepository
from repositories.model_repository import ModelRepository


class ModelService:
    def __init__(self) -> None:
        self.behavior_repository = BehaviorRepository()
        self.model_repository = ModelRepository()

    def active_model(self, model_name: str) -> Optional[Dict]:
        return self.model_repository.active_model(model_name)

    def list_versions(self, model_name: str) -> List[Dict]:
        return self.model_repository.list_versions(model_name)

    def save_version(self, model_name: str, version_tag: str, storage_path: str, metrics: Dict, note: str) -> Dict:
        return self.model_repository.save_version(model_name, version_tag, storage_path, metrics, note)

    def rollback(self, model_name: str, version_tag: str) -> bool:
        if not self.model_repository.version_exists(model_name, version_tag):
            return False
        self.model_repository.activate_version(model_name, version_tag)
        return True

    def delete_version(self, model_name: str, version_tag: str) -> None:
        self.model_repository.delete_version(model_name, version_tag)

    def feedback_rows(self) -> List[tuple]:
        return self.behavior_repository.list_feedback_rows()

    def movie_exposure_map(self) -> Dict[int, int]:
        return self.behavior_repository.get_movie_exposure_map()

    def insert_experience(
        self,
        user_email: str,
        movie_id: int,
        behavior_type: int,
        reward: float,
        old_prob: Optional[float],
        payload: Dict,
    ) -> None:
        self.behavior_repository.insert_experience(user_email, movie_id, behavior_type, reward, old_prob, payload)

    def pending_experience_count(self) -> int:
        return self.behavior_repository.pending_experience_count()

    def list_pending_experiences(self, limit: int) -> List[tuple]:
        return self.behavior_repository.list_pending_experiences(limit)

    def mark_experiences_processed(self, ids: Sequence[int], model_version: str) -> None:
        self.behavior_repository.mark_experiences_processed(ids, model_version)
