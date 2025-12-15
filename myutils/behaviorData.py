from typing import Dict, List, Optional

from .rl.features import get_state_vector, invalidate_user_feature_cache
from .rl.local_ppo import record_online_feedback
from services.behavior_service import BehaviorService

behavior_service = BehaviorService()


def add_behavior(user_email: str, movie_id: int, behavior_type: int, score: Optional[float] = None) -> str:
    pre_state = get_state_vector(user_email).tolist()
    behavior_service.add_behavior(user_email, movie_id, behavior_type, score)

    invalidate_user_feature_cache(user_email)
    next_state = get_state_vector(user_email, force_refresh=True).tolist()
    record_online_feedback(
        user_email=user_email,
        movie_id=movie_id,
        behavior_type=behavior_type,
        score=score,
        state_vector=pre_state,
        next_state_vector=next_state,
    )
    return "success"


def get_user_behavior(user_email: str) -> List[Dict]:
    return behavior_service.get_user_behavior(user_email)


def get_behavior_snapshot(user_email: Optional[str], movie_id: int) -> Dict:
    return behavior_service.get_behavior_snapshot(user_email, movie_id)
