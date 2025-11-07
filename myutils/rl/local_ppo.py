from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Linear(state_dim, action_dim)
        self.critic = nn.Linear(state_dim, 1)

    def forward(self, state, action_features):
        emb = torch.tanh(self.actor(state))
        logits = (action_features @ emb) / torch.sqrt(torch.tensor(action_features.shape[1], dtype=torch.float32))
        logits = logits - torch.max(logits)
        probs = torch.softmax(logits, dim=0)
        value = self.critic(state).squeeze()
        return probs, value, emb


class LocalPPOReranker:
    def __init__(self, state_dim=STATE_VECTOR_DIM, action_dim=MOVIE_FEATURE_DIM):
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=PPO_LR)

    def predict(self, state_np, action_features_np):
        with torch.no_grad():
            state = torch.tensor(state_np, dtype=torch.float32).to(device)
            action_features = torch.tensor(action_features_np, dtype=torch.float32).to(device)
            probs, value, emb = self.model(state, action_features)
        return probs.cpu().numpy(), value.item(), emb.cpu().numpy()

    def action_probabilities(self, state_np, feature_map: Dict[int, np.ndarray]) -> Dict[int, float]:
        action_ids, action_features = build_action_matrix(feature_map)
        if len(action_ids) == 0:
            return {}
        probs, _, _ = self.predict(state_np, action_features)
        return {mid: float(probs[i]) for i, mid in enumerate(action_ids)}

    def rerank_movies(
        self, user_email: str, candidates: Sequence[Dict], top_n: int,
        feature_map: Optional[Dict[int, np.ndarray]] = None
    ) -> List[Dict]:
        feature_map = feature_map or get_movie_feature_map()
        state = get_state_vector(user_email)
        prob_map = self.action_probabilities(state, feature_map)
        exposure_map = get_movie_exposure_map()
        enriched = []

        for movie in candidates:
            mid = int(movie["id"])
            item = dict(movie)
            item["policy_score"] = prob_map.get(mid, 0.0)
            item["coverage_bonus"] = coverage_bonus(mid, exposure_map)
            item["rl_score"] = item["policy_score"] + RL_COVERAGE_WEIGHT * item["coverage_bonus"]
            enriched.append(item)

        remaining = sorted(enriched, key=lambda x: x.get("rl_score", 0.0), reverse=True)
        selected = []
        rng = torch.Generator().manual_seed(42)

        while remaining and len(selected) < top_n:
            if torch.rand(1, generator=rng).item() < RL_EPSILON and len(remaining) > 1:
                weights = np.array([max(x.get("coverage_bonus", 0.0), 0.01) for x in remaining], dtype=np.float32)
                weights /= weights.sum()
                chosen_idx = int(torch.multinomial(torch.tensor(weights), 1, generator=rng).item())
            else:
                scored = []
                for idx, item in enumerate(remaining):
                    feat = feature_map.get(int(item["id"]))
                    div = diversity_bonus(feat, [feature_map.get(int(r["id"])) for r in selected])
                    scored.append((idx, item["rl_score"] + RL_DIVERSITY_WEIGHT * div, div))
                chosen_idx, total_score, div = max(scored, key=lambda x: x[1])
                remaining[chosen_idx]["diversity_bonus"] = div
                remaining[chosen_idx]["rl_score"] = total_score

            chosen = remaining.pop(chosen_idx)
            chosen["exploration_used"] = chosen_idx != 0
            reason = chosen.get("reason", "")
            chosen["reason"] = (
                f"{reason}；PPO {chosen.get('policy_score',0):.3f}｜多样性 {chosen.get('diversity_bonus',0):.3f}｜覆盖率 {chosen.get('coverage_bonus',0):.3f}"
            ).strip("；")
            selected.append(chosen)
        return selected[:top_n]

    def train(
        self, experiences: Sequence[Dict], feature_map: Dict[int, np.ndarray],
        epochs=PPO_EPOCHS, lr=PPO_LR, clip_eps=PPO_CLIP_EPSILON
    ) -> TrainingMetrics:
        action_ids, action_features_np = build_action_matrix(feature_map)
        movie_to_idx = {mid: i for i, mid in enumerate(action_ids)}
        rewards = []
        aux_rewards = []

        action_features = torch.tensor(action_features_np, dtype=torch.float32).to(device)
        self.model.train()

        for g in self.optimizer.param_groups:
            g["lr"] = lr

        for _ in range(max(epochs, 1)):
            for exp in experiences:
                aid = movie_to_idx.get(exp["movie_id"])
                if aid is None:
                    continue

                state = torch.tensor(exp["state"], dtype=torch.float32).to(device)
                aux = float(exp.get("aux_reward", 0))
                reward = float(exp["reward"]) + aux
                rewards.append(reward)
                aux_rewards.append(aux)

                old_prob = float(exp.get("old_prob", 1/len(action_ids)))
                old_prob = max(old_prob, 1e-8)

                probs, value, _ = self.model(state, action_features)
                curr_prob = probs[aid].clamp_min(1e-8)
                advantage = reward - value.item()
                ratio = (curr_prob / old_prob).float()
                clip_low = 1 - clip_eps
                clip_high = 1 + clip_eps

                clipped = (advantage >= 0 and ratio > clip_high) or (advantage < 0 and ratio < clip_low)

                if not clipped:
                    obj = -advantage * ratio * torch.log(curr_prob)
                else:
                    obj = torch.tensor(0.0, device=device)

                value_loss = (value - reward) ** 2
                loss = obj + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return TrainingMetrics(
            batch_size=len(experiences),
            avg_reward=np.mean(rewards or [0]),
            positive_rate=np.mean(np.array(rewards or [0]) > 0),
            epochs=epochs,
            avg_aux_reward=np.mean(aux_rewards or [0])
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: Path):
        model = cls()
        model.model.load_state_dict(torch.load(path, map_location=device))
        return model

def build_action_matrix(feature_map: Dict[int, np.ndarray]):
    action_ids = sorted(feature_map.keys())
    if not action_ids:
        return [], np.zeros((0, MOVIE_FEATURE_DIM))
    action_features = np.vstack([feature_map[mid] for mid in action_ids])
    return action_ids, action_features

def reward_from_behavior(behavior_type: int, score: Optional[float]) -> float:
    if behavior_type == 1 and score is not None:
        return round((float(score) - 5.0) / 5.0, 4)
    if behavior_type == 2: return 0.35
    if behavior_type == 3: return 0.15
    return 0.0

def get_movie_exposure_map() -> Dict[int, int]:
    return model_service.movie_exposure_map()

def coverage_bonus(movie_id: int, exposure_map=None) -> float:
    exposure_map = exposure_map or get_movie_exposure_map()
    if not exposure_map: return 1.0
    max_exp = max(exposure_map.values()) or 1
    exp = exposure_map.get(movie_id, 0)
    return round(1.0 - exp/max_exp, 4)

def diversity_bonus(candidate_feat, selected_feats):
    if candidate_feat is None or not selected_feats: return 1.0
    sims = []
    ca_norm = np.linalg.norm(candidate_feat) or 1
    for sf in selected_feats:
        if sf is None: continue
        sf_norm = np.linalg.norm(sf) or 1
        sims.append(np.dot(candidate_feat, sf)/(ca_norm*sf_norm))
    if not sims: return 1.0
    return round(max(0.0, 1.0 - float(np.mean(sims))), 4)

def _cleanup_old_model_versions():
    versions = list_model_versions()
    if len(versions) <= RL_KEEP_MODEL_VERSIONS: return
    for v in versions[RL_KEEP_MODEL_VERSIONS:]:
        try:
            Path(v["storage_path"]).unlink(missing_ok=True)
        except: pass
        model_service.delete_version(MODEL_NAME, v["version_tag"])

def _now_version_tag():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _active_model_row():
    return model_service.active_model(MODEL_NAME)

def list_model_versions():
    return model_service.list_versions(MODEL_NAME)

def load_active_model() -> Optional[LocalPPOReranker]:
    row = _active_model_row()
    if not row: return None
    p = Path(row["storage_path"])
    if not p.exists(): return None
    return LocalPPOReranker.load(p)

def _save_model_version(model: LocalPPOReranker, metrics, note):
    tag = _now_version_tag()
    path = MODEL_DIR / f"{tag}.pt"
    model.save(path)
    saved = model_service.save_version(MODEL_NAME, tag, str(path), metrics, note)
    _cleanup_old_model_versions()
    return saved

def rollback_model_version(tag):
    return model_service.rollback(MODEL_NAME, tag)

def get_pending_experience_count():
    return model_service.pending_experience_count()

def _rows_to_experiences(rows):
    exps = []
    for r in rows:
        payload = json.loads(r[5] or "{}")
        state = payload.get("state")
        if not state: continue
        exps.append({
            "id": int(r[0]),
            "user_email": r[1],
            "movie_id": int(r[2]),
            "reward": float(r[3]),
            "old_prob": r[4],
            "state": state,
            "aux_reward": float(payload.get("aux_reward", 0)),
        })
    return exps

def ensure_bootstrap_model():
    if not RL_ENABLED: return None
    if _active_model_row(): return
    rows = model_service.feedback_rows()
    if len(rows) < RL_MIN_FEEDBACK: return None
    fm = get_movie_feature_map()
    if not fm: return None

    exps = []
    total = len(get_all_movie_ids()) or 1
    for u, mid, bt, s in rows:
        exps.append({
            "user_email": u,
            "movie_id": int(mid),
            "reward": reward_from_behavior(int(bt), s),
            "old_prob": 1/total,
            "state": get_state_vector(u).tolist(),
        })

    model = LocalPPOReranker()
    metrics = model.train(exps, fm).to_dict()
    return _save_model_version(model, metrics, "bootstrap")

def record_online_feedback(user_email, movie_id, behavior_type, score, state_vec=None, next_state_vec=None):
    if not RL_ENABLED: return
    fm = get_movie_feature_map()
    if movie_id not in fm: return

    model = load_active_model()
    state = state_vec if state_vec is not None else get_state_vector(user_email)
    old_prob = model.action_probabilities(state, fm).get(movie_id) if model else None

    aux = RL_COVERAGE_WEIGHT * coverage_bonus(movie_id)
    payload = {
        "state": state.tolist() if hasattr(state, "tolist") else state,
        "next_state": list(next_state_vec) if next_state_vec else None,
        "score": score,
        "aux_reward": aux,
    }
    model_service.insert_experience(
        user_email=user_email, movie_id=movie_id, behavior_type=behavior_type,
        reward=reward_from_behavior(behavior_type, score), old_prob=old_prob, payload=payload
    )
    train_pending_batch_if_ready()

def train_pending_batch_if_ready(force=False):
    if not RL_ENABLED: return None
    cnt = get_pending_experience_count()
    if not force and cnt < RL_BATCH_SIZE: return None
    rows = model_service.list_pending_experiences(max(RL_BATCH_SIZE, cnt))
    exps = _rows_to_experiences(rows)
    if not exps: return None
    fm = get_movie_feature_map()
    model = load_active_model() or LocalPPOReranker()
    metrics = model.train(exps, fm).to_dict()
    saved = _save_model_version(model, metrics, f"batch {len(exps)}")
    model_service.mark_experiences_processed([e["id"] for e in exps], saved["version_tag"])
    return saved

def rerank_with_local_ppo(user_email, candidates, top_n):
    if not RL_ENABLED or not user_email or not candidates:
        return list(candidates)[:top_n]
    ensure_bootstrap_model()
    model = load_active_model()
    if not model: return list(candidates)[:top_n]
    fm = get_movie_feature_map()
    available = [m for m in candidates if int(m["id"]) in fm]
    if not available: return list(candidates)[:top_n]
    reranked = model.rerank_movies(user_email, available, top_n, fm)
    used = {x["id"] for x in reranked}
    for m in candidates:
        if m["id"] not in used and len(reranked) < top_n:
            reranked.append(m)
    return reranked[:top_n]

def status_payload():
    return {
        "enabled": RL_ENABLED, "batch_size": RL_BATCH_SIZE, "epsilon": RL_EPSILON,
        "diversity_weight": RL_DIVERSITY_WEIGHT, "coverage_weight": RL_COVERAGE_WEIGHT,
        "pending_experiences": get_pending_experience_count(),
        "active_model": _active_model_row(), "versions": list_model_versions()
    }

def _build_cli():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="command", required=True)
    sp.add_parser("status")
    sp.add_parser("bootstrap")
    tr = sp.add_parser("train")
    tr.add_argument("--force", action="store_true")
    rb = sp.add_parser("rollback")
    rb.add_argument("version_tag")
    sp.add_parser("list")
    return p

def main():
    args = _build_cli().parse_args()
    if args.command == "status": print(json.dumps(status_payload(), indent=2, ensure_ascii=False))
    elif args.command == "bootstrap": print(json.dumps(ensure_bootstrap_model(), indent=2, ensure_ascii=False))
    elif args.command == "train": print(json.dumps(train_pending_batch_if_ready(force=args.force), indent=2, ensure_ascii=False))
    elif args.command == "rollback": print(json.dumps({"ok": rollback_model_version(args.version_tag)}, indent=2, ensure_ascii=False))
    elif args.command == "list": print(json.dumps(list_model_versions(), indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())