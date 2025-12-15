from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .features import MOVIE_FEATURE_DIM, STATE_VECTOR_DIM, get_movie_feature_map, get_state_vector
from ..query import get_all_movie_ids
from services.model_service import ModelService


MODEL_NAME = "local_ppo_reranker"
RL_ENABLED = os.getenv("MOVIE_RL_ENABLED", "1") != "0"
RL_BATCH_SIZE = int(os.getenv("MOVIE_RL_BATCH_SIZE", "100"))
RL_MIN_FEEDBACK = int(os.getenv("MOVIE_RL_MIN_FEEDBACK", "5"))
PPO_EPOCHS = int(os.getenv("MOVIE_PPO_EPOCHS", "6"))
PPO_LR = float(os.getenv("MOVIE_PPO_LR", "0.03"))
PPO_CLIP_EPSILON = float(os.getenv("MOVIE_PPO_CLIP", "0.2"))
RL_EPSILON = float(os.getenv("MOVIE_RL_EPSILON", "0.1"))
RL_DIVERSITY_WEIGHT = float(os.getenv("MOVIE_RL_DIVERSITY_WEIGHT", "0.18"))
RL_COVERAGE_WEIGHT = float(os.getenv("MOVIE_RL_COVERAGE_WEIGHT", "0.12"))
RL_KEEP_MODEL_VERSIONS = int(os.getenv("MOVIE_KEEP_MODEL_VERSIONS", "6"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MOVIE_MODEL_DIR", PROJECT_ROOT / "artifacts" / "rl_models" / "ppo"))
model_service = ModelService()


@dataclass
class TrainingMetrics:
    batch_size: int
    avg_reward: float
    positive_rate: float
    epochs: int
    avg_aux_reward: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "batch_size": self.batch_size,
            "avg_reward": round(self.avg_reward, 4),
            "positive_rate": round(self.positive_rate, 4),
            "epochs": self.epochs,
            "avg_aux_reward": round(self.avg_aux_reward, 4),
        }


class LocalPPOReranker:
    def __init__(
        self,
        state_dim: int = STATE_VECTOR_DIM,
        action_dim: int = MOVIE_FEATURE_DIM,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.W_actor = rng.normal(0, 0.05, size=(state_dim, action_dim))
        self.b_actor = np.zeros(action_dim, dtype=float)
        self.W_value = np.zeros(state_dim, dtype=float)
        self.b_value = 0.0

    def _state_embedding(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(state @ self.W_actor + self.b_actor)

    def predict(self, state: np.ndarray, action_features: np.ndarray):
        embedding = self._state_embedding(state)
        logits = (action_features @ embedding) / np.sqrt(max(action_features.shape[1], 1))
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= max(np.sum(probs), 1e-8)
        value = float(state @ self.W_value + self.b_value)
        return probs, value, embedding

    def action_probabilities(self, state: np.ndarray, feature_map: Dict[int, np.ndarray]) -> Dict[int, float]:
        action_ids, action_features = build_action_matrix(feature_map)
        if len(action_ids) == 0:
            return {}
        probs, _, _ = self.predict(state, action_features)
        return {movie_id: float(probs[idx]) for idx, movie_id in enumerate(action_ids)}

    def rerank_movies(
        self,
        user_email: str,
        candidates: Sequence[Dict],
        top_n: int,
        feature_map: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[Dict]:
        feature_map = feature_map or get_movie_feature_map()
        state = get_state_vector(user_email)
        prob_map = self.action_probabilities(state, feature_map)
        exposure_map = get_movie_exposure_map()
        enriched = []
        for movie in candidates:
            movie_id = int(movie["id"])
            item = dict(movie)
            item["policy_score"] = prob_map.get(movie_id, 0.0)
            item["coverage_bonus"] = coverage_bonus(movie_id, exposure_map)
            item["rl_score"] = item["policy_score"] + RL_COVERAGE_WEIGHT * item["coverage_bonus"]
            enriched.append(item)

        remaining = sorted(enriched, key=lambda item: item.get("rl_score", 0.0), reverse=True)
        selected = []
        rng = np.random.default_rng()
        while remaining and len(selected) < top_n:
            if rng.random() < RL_EPSILON and len(remaining) > 1:
                weights = np.array([max(item.get("coverage_bonus", 0.0), 0.01) for item in remaining], dtype=float)
                weights /= weights.sum()
                chosen_idx = int(rng.choice(len(remaining), p=weights))
            else:
                scored_candidates = []
                for idx, item in enumerate(remaining):
                    movie_feature = feature_map.get(int(item["id"]))
                    diversity = diversity_bonus(movie_feature, [feature_map.get(int(row["id"])) for row in selected])
                    total_score = item["rl_score"] + RL_DIVERSITY_WEIGHT * diversity
                    scored_candidates.append((idx, total_score, diversity))
                chosen_idx, total_score, diversity = max(scored_candidates, key=lambda row: row[1])
                remaining[chosen_idx]["diversity_bonus"] = diversity
                remaining[chosen_idx]["rl_score"] = total_score

            chosen = remaining.pop(chosen_idx)
            chosen["exploration_used"] = chosen_idx != 0
            reason = chosen.get("reason", "")
            chosen["reason"] = (
                f"{reason}；PPO {chosen.get('policy_score', 0.0):.3f}"
                f"｜多样性 {chosen.get('diversity_bonus', 0.0):.3f}"
                f"｜覆盖率 {chosen.get('coverage_bonus', 0.0):.3f}"
            ).strip("；")
            selected.append(chosen)
        return selected[:top_n]

    def train(
        self,
        experiences: Sequence[Dict],
        feature_map: Dict[int, np.ndarray],
        epochs: int = PPO_EPOCHS,
        learning_rate: float = PPO_LR,
        clip_epsilon: float = PPO_CLIP_EPSILON,
    ) -> TrainingMetrics:
        action_ids, action_features = build_action_matrix(feature_map)
        movie_to_index = {movie_id: idx for idx, movie_id in enumerate(action_ids)}
        rewards = []
        aux_rewards = []

        for _ in range(max(epochs, 1)):
            for exp in experiences:
                action_idx = movie_to_index.get(exp["movie_id"])
                if action_idx is None:
                    continue

                state = np.array(exp["state"], dtype=float)
                aux_reward = float(exp.get("aux_reward") or 0.0)
                reward = float(exp["reward"]) + aux_reward
                rewards.append(reward)
                aux_rewards.append(aux_reward)
                probs, value, embedding = self.predict(state, action_features)
                current_prob = max(float(probs[action_idx]), 1e-8)
                old_prob = max(float(exp.get("old_prob") or (1.0 / max(len(action_ids), 1))), 1e-8)
                advantage = reward - value
                ratio = current_prob / old_prob
                clip_low = 1.0 - clip_epsilon
                clip_high = 1.0 + clip_epsilon
                clipped = (advantage >= 0 and ratio > clip_high) or (advantage < 0 and ratio < clip_low)

                if not clipped:
                    expected_feature = probs @ action_features
                    grad_log_prob = action_features[action_idx] - expected_feature
                    grad_pre_activation = grad_log_prob * (1.0 - embedding ** 2)
                    actor_scale = -advantage * ratio
                    self.W_actor -= learning_rate * np.outer(state, actor_scale * grad_pre_activation)
                    self.b_actor -= learning_rate * actor_scale * grad_pre_activation

                value_error = value - reward
                self.W_value -= learning_rate * value_error * state
                self.b_value -= learning_rate * value_error

        rewards_array = np.array(rewards or [0.0], dtype=float)
        return TrainingMetrics(
            batch_size=len(experiences),
            avg_reward=float(rewards_array.mean()),
            positive_rate=float((rewards_array > 0).mean()),
            epochs=max(epochs, 1),
            avg_aux_reward=float(np.array(aux_rewards or [0.0], dtype=float).mean()),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W_actor=self.W_actor,
            b_actor=self.b_actor,
            W_value=self.W_value,
            b_value=np.array([self.b_value], dtype=float),
        )

    @classmethod
    def load(cls, path: Path) -> "LocalPPOReranker":
        payload = np.load(path)
        model = cls()
        model.W_actor = payload["W_actor"]
        model.b_actor = payload["b_actor"]
        model.W_value = payload["W_value"]
        model.b_value = float(payload["b_value"][0])
        return model


def build_action_matrix(feature_map: Dict[int, np.ndarray]):
    action_ids = sorted(feature_map.keys())
    action_features = np.vstack([feature_map[movie_id] for movie_id in action_ids]) if action_ids else np.zeros((0, MOVIE_FEATURE_DIM))
    return action_ids, action_features


def reward_from_behavior(behavior_type: int, score: Optional[float]) -> float:
    if behavior_type == 1 and score is not None:
        return round((float(score) - 5.0) / 5.0, 4)
    if behavior_type == 2:
        return 0.35
    if behavior_type == 3:
        return 0.15
    return 0.0


def get_movie_exposure_map() -> Dict[int, int]:
    return model_service.movie_exposure_map()


def coverage_bonus(movie_id: int, exposure_map: Optional[Dict[int, int]] = None) -> float:
    exposure_map = exposure_map or get_movie_exposure_map()
    if not exposure_map:
        return 1.0
    max_exposure = max(exposure_map.values()) or 1
    exposure = exposure_map.get(movie_id, 0)
    return round(1.0 - (exposure / max_exposure), 4)


def diversity_bonus(candidate_feature: Optional[np.ndarray], selected_features: Sequence[Optional[np.ndarray]]) -> float:
    if candidate_feature is None or not selected_features:
        return 1.0
    similarities = []
    candidate_norm = np.linalg.norm(candidate_feature) or 1.0
    for feature in selected_features:
        if feature is None:
            continue
        feature_norm = np.linalg.norm(feature) or 1.0
        similarities.append(float(np.dot(candidate_feature, feature) / (candidate_norm * feature_norm)))
    if not similarities:
        return 1.0
    return round(max(0.0, 1.0 - float(np.mean(similarities))), 4)


def _cleanup_old_model_versions() -> None:
    versions = list_model_versions()
    if len(versions) <= RL_KEEP_MODEL_VERSIONS:
        return
    removable = versions[RL_KEEP_MODEL_VERSIONS:]
    for version in removable:
        try:
            Path(version["storage_path"]).unlink(missing_ok=True)
        except Exception:
            pass
        model_service.delete_version(MODEL_NAME, version["version_tag"])


def _now_version_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _active_model_row() -> Optional[Dict]:
    return model_service.active_model(MODEL_NAME)


def list_model_versions() -> List[Dict]:
    return model_service.list_versions(MODEL_NAME)


def load_active_model() -> Optional[LocalPPOReranker]:
    row = _active_model_row()
    if not row:
        return None
    path = Path(row["storage_path"])
    if not path.exists():
        return None
    return LocalPPOReranker.load(path)


def _save_model_version(model: LocalPPOReranker, metrics: Dict, note: str) -> Dict:
    version_tag = _now_version_tag()
    path = MODEL_DIR / f"{version_tag}.npz"
    model.save(path)
    saved = model_service.save_version(MODEL_NAME, version_tag, str(path), metrics, note)
    _cleanup_old_model_versions()
    return saved


def rollback_model_version(version_tag: str) -> bool:
    return model_service.rollback(MODEL_NAME, version_tag)


def get_pending_experience_count() -> int:
    return model_service.pending_experience_count()


def _rows_to_experiences(rows: Sequence[Sequence]) -> List[Dict]:
    experiences = []
    for row in rows:
        payload = json.loads(row[5] or "{}")
        state = payload.get("state")
        if not state:
            continue
        experiences.append(
            {
                "id": int(row[0]),
                "user_email": row[1],
                "movie_id": int(row[2]),
                "reward": float(row[3]),
                "old_prob": row[4],
                "state": state,
                "aux_reward": float(payload.get("aux_reward") or 0.0),
            }
        )
    return experiences


def ensure_bootstrap_model() -> Optional[Dict]:
    if not RL_ENABLED:
        return None
    active = _active_model_row()
    if active:
        return active

    feedback_rows = model_service.feedback_rows()
    if len(feedback_rows) < RL_MIN_FEEDBACK:
        return None

    feature_map = get_movie_feature_map()
    if not feature_map:
        return None

    action_count = max(len(get_all_movie_ids()), 1)
    experiences = []
    for user_email, movie_id, behavior_type, score in feedback_rows:
        state = get_state_vector(user_email).tolist()
        experiences.append(
            {
                "user_email": user_email,
                "movie_id": int(movie_id),
                "reward": reward_from_behavior(int(behavior_type), score),
                "old_prob": 1.0 / action_count,
                "state": state,
            }
        )

    model = LocalPPOReranker()
    metrics = model.train(experiences, feature_map).to_dict()
    return _save_model_version(model, metrics, "bootstrap from historical feedback")


def record_online_feedback(
    user_email: str,
    movie_id: int,
    behavior_type: int,
    score: Optional[float],
    state_vector: Optional[Sequence[float]] = None,
    next_state_vector: Optional[Sequence[float]] = None,
) -> None:
    if not RL_ENABLED:
        return

    feature_map = get_movie_feature_map()
    if movie_id not in feature_map:
        return

    model = load_active_model()
    state_array = np.array(state_vector, dtype=float) if state_vector is not None else get_state_vector(user_email)
    old_prob = None
    if model is not None:
        old_prob = model.action_probabilities(state_array, feature_map).get(movie_id)

    aux_reward = RL_COVERAGE_WEIGHT * coverage_bonus(movie_id)
    payload = {
        "state": state_array.tolist(),
        "next_state": list(next_state_vector) if next_state_vector is not None else None,
        "score": score,
        "aux_reward": aux_reward,
    }
    model_service.insert_experience(
        user_email=user_email,
        movie_id=movie_id,
        behavior_type=behavior_type,
        reward=reward_from_behavior(behavior_type, score),
        old_prob=old_prob,
        payload=payload,
    )
    train_pending_batch_if_ready()


def train_pending_batch_if_ready(force: bool = False) -> Optional[Dict]:
    if not RL_ENABLED:
        return None

    pending_count = get_pending_experience_count()
    if not force and pending_count < RL_BATCH_SIZE:
        return None

    rows = model_service.list_pending_experiences(max(RL_BATCH_SIZE, pending_count))
    experiences = _rows_to_experiences(rows)
    if not experiences:
        return None

    feature_map = get_movie_feature_map()
    model = load_active_model() or LocalPPOReranker()
    metrics = model.train(experiences, feature_map).to_dict()
    saved = _save_model_version(model, metrics, f"online batch update ({len(experiences)} samples)")
    processed_ids = [item["id"] for item in experiences]
    model_service.mark_experiences_processed(processed_ids, saved["version_tag"])
    return saved


def rerank_with_local_ppo(user_email: Optional[str], candidates: Sequence[Dict], top_n: int) -> List[Dict]:
    if not RL_ENABLED or not user_email or not candidates:
        return list(candidates)[:top_n]

    ensure_bootstrap_model()
    model = load_active_model()
    if model is None:
        return list(candidates)[:top_n]

    feature_map = get_movie_feature_map()
    available_candidates = [movie for movie in candidates if int(movie["id"]) in feature_map]
    if not available_candidates:
        return list(candidates)[:top_n]

    reranked = model.rerank_movies(user_email, available_candidates, top_n, feature_map)
    used_ids = {item["id"] for item in reranked}
    for movie in candidates:
        if movie["id"] not in used_ids and len(reranked) < top_n:
            reranked.append(movie)
    return reranked[:top_n]


def status_payload() -> Dict:
    active = _active_model_row()
    return {
        "enabled": RL_ENABLED,
        "batch_size": RL_BATCH_SIZE,
        "epsilon": RL_EPSILON,
        "diversity_weight": RL_DIVERSITY_WEIGHT,
        "coverage_weight": RL_COVERAGE_WEIGHT,
        "pending_experiences": get_pending_experience_count(),
        "active_model": active,
        "versions": list_model_versions(),
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local PPO recommendation models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show active PPO model and pending experience count.")
    subparsers.add_parser("bootstrap", help="Bootstrap a PPO model from historical feedback.")
    train_parser = subparsers.add_parser("train", help="Train on pending online experiences.")
    train_parser.add_argument("--force", action="store_true", help="Train even if batch size is not reached.")
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a historical model version.")
    rollback_parser.add_argument("version_tag", help="Version tag to activate.")
    subparsers.add_parser("list", help="List all saved model versions.")
    return parser


def main() -> int:
    parser = _build_cli()
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(status_payload(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "bootstrap":
        print(json.dumps(ensure_bootstrap_model(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        print(json.dumps(train_pending_batch_if_ready(force=args.force), ensure_ascii=False, indent=2))
        return 0
    if args.command == "rollback":
        result = rollback_model_version(args.version_tag)
        print(json.dumps({"ok": result, "version_tag": args.version_tag}, ensure_ascii=False, indent=2))
        return 0 if result else 1
    if args.command == "list":
        print(json.dumps(list_model_versions(), ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
