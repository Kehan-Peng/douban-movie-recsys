from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from .query import get_movie_data
from .recommend import _build_similarity_scores, _hot_movies
from .recommender.semantic_embeddings import (
    score_cf_semantic,
    score_content_semantic,
)
from .rl.features import get_movie_feature_map
from .rl.local_ppo import LocalPPOReranker, load_active_model
from services.behavior_service import BehaviorService


behavior_service = BehaviorService()


def _movie_catalog() -> Dict[int, Dict]:
    catalog = {}
    for movie in (get_movie_data() or []):
        normalized = dict(movie)
        normalized["types"] = movie.get("types_list") or movie.get("genres") or []
        normalized["directors"] = movie.get("directors_list") or []
        normalized["country"] = movie.get("country_list") or []
        normalized["summary"] = movie.get("summary") or movie.get("description") or ""
        catalog[int(movie["id"])] = normalized
    return catalog


def _load_ratings() -> Dict[str, List[Dict]]:
    ratings = defaultdict(list)
    for row in behavior_service.get_rating_events():
        ratings[str(row["user_email"])].append(
            {
                "movie_id": int(row["movie_id"]),
                "score": float(row["score"] or 0.0),
                "create_time": row["create_time"],
            }
        )
    return ratings


def _build_eval_samples(all_ratings: Dict[str, List[Dict]]) -> List[Dict]:
    samples = []
    for user_email, history in all_ratings.items():
        positive_history = [row for row in history if row["score"] >= 8]
        if len(positive_history) < 2:
            continue
        target = positive_history[-1]
        train_rows = [row for row in history if row is not target]
        train_preferences = {row["movie_id"]: row["score"] for row in train_rows}
        samples.append(
            {
                "user_email": user_email,
                "target_movie_id": target["movie_id"],
                "train_rows": train_rows,
                "train_preferences": train_preferences,
            }
        )
    return samples


def _pearson_similarity(left: Dict[int, float], right: Dict[int, float]) -> float:
    common = set(left) & set(right)
    if len(common) < 2:
        return 0.0
    left_mean = sum(left[item] for item in common) / len(common)
    right_mean = sum(right[item] for item in common) / len(common)
    numerator = sum((left[item] - left_mean) * (right[item] - right_mean) for item in common)
    left_norm = math.sqrt(sum((left[item] - left_mean) ** 2 for item in common))
    right_norm = math.sqrt(sum((right[item] - right_mean) ** 2 for item in common))
    denominator = left_norm * right_norm
    return numerator / denominator if denominator else 0.0


def _baseline_content_scores(sample: Dict, movies: Dict[int, Dict]) -> Dict[int, float]:
    liked_movie_ids = [movie_id for movie_id, score in sample["train_preferences"].items() if score >= 8]
    if not liked_movie_ids:
        return {}
    movie_records = list(movies.values())
    scores = _build_similarity_scores(movie_records, liked_movie_ids)
    for watched_id in sample["train_preferences"]:
        scores.pop(watched_id, None)
    return scores


def _baseline_cf_scores(sample: Dict, all_preferences: Dict[str, Dict[int, float]]) -> Dict[int, float]:
    user_email = sample["user_email"]
    target_preferences = sample["train_preferences"]
    if len(target_preferences) < 2:
        return {}
    similarities = {}
    for other_user, other_preferences in all_preferences.items():
        if other_user == user_email:
            continue
        similarity = _pearson_similarity(target_preferences, other_preferences)
        if similarity > 0:
            similarities[other_user] = similarity
    scores = defaultdict(float)
    watched_ids = set(target_preferences.keys())
    for other_user, similarity in similarities.items():
        for movie_id, score in all_preferences[other_user].items():
            if movie_id in watched_ids or score < 8:
                continue
            scores[movie_id] += similarity * score
    return scores


def _merge_score_maps(score_maps: Sequence[Dict[int, float]], weights: Sequence[float]) -> Dict[int, float]:
    merged = defaultdict(float)
    for score_map, weight in zip(score_maps, weights):
        for movie_id, score in score_map.items():
            merged[movie_id] += weight * score
    return merged


def _rank_ids(score_map: Dict[int, float], top_k: int) -> List[int]:
    return [movie_id for movie_id, _ in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:top_k]]


def _state_from_preferences(preferences: Dict[int, float], feature_map: Dict[int, np.ndarray]) -> np.ndarray:
    weighted_vectors = []
    weights = []
    for movie_id, score in preferences.items():
        vector = feature_map.get(int(movie_id))
        if vector is None:
            continue
        weighted_vectors.append(vector)
        weights.append(max(float(score) / 10.0, 0.1))
    if not weighted_vectors:
        feature_dim = next(iter(feature_map.values())).shape[0] if feature_map else 0
        return np.zeros(feature_dim * 2, dtype=float)
    preference = np.average(np.vstack(weighted_vectors), axis=0, weights=np.array(weights, dtype=float))
    return np.concatenate([preference, preference])


def _ppo_rerank_ids(candidate_ids: List[int], preferences: Dict[int, float], feature_map: Dict[int, np.ndarray], top_k: int) -> List[int]:
    if not candidate_ids:
        return []
    model = load_active_model() or LocalPPOReranker()
    state = _state_from_preferences(preferences, feature_map)
    candidate_feature_map = {movie_id: feature_map[movie_id] for movie_id in candidate_ids if movie_id in feature_map}
    probabilities = model.action_probabilities(state, candidate_feature_map)
    return [movie_id for movie_id in sorted(candidate_ids, key=lambda movie_id: probabilities.get(movie_id, 0.0), reverse=True)[:top_k]]


def _precision_at_k(recommended_ids: Sequence[int], target_id: int, k: int) -> float:
    return (1.0 if target_id in recommended_ids[:k] else 0.0) / max(k, 1)


def _recall_at_k(recommended_ids: Sequence[int], target_id: int, k: int) -> float:
    return 1.0 if target_id in recommended_ids[:k] else 0.0


def _ndcg_at_k(recommended_ids: Sequence[int], target_id: int, k: int) -> float:
    for index, movie_id in enumerate(recommended_ids[:k], start=1):
        if movie_id == target_id:
            return 1.0 / math.log2(index + 1)
    return 0.0


def _list_diversity(movie_ids: Sequence[int], feature_map: Dict[int, np.ndarray]) -> float:
    if len(movie_ids) < 2:
        return 0.0
    distances = []
    for idx, movie_id in enumerate(movie_ids):
        current = feature_map.get(movie_id)
        if current is None:
            continue
        for other_id in movie_ids[idx + 1 :]:
            other = feature_map.get(other_id)
            if other is None:
                continue
            current_norm = np.linalg.norm(current) or 1.0
            other_norm = np.linalg.norm(other) or 1.0
            similarity = float(np.dot(current, other) / (current_norm * other_norm))
            distances.append(max(0.0, 1.0 - similarity))
    return round(float(np.mean(distances)) if distances else 0.0, 4)


def evaluate_recommenders(top_k: int = 5) -> Dict:
    movies = _movie_catalog()
    all_ratings = _load_ratings()
    all_preferences = {user_email: {row["movie_id"]: row["score"] for row in rows} for user_email, rows in all_ratings.items()}
    samples = _build_eval_samples(all_ratings)
    feature_map = get_movie_feature_map()
    summary = {"sample_users": len(samples), "top_k": top_k, "metrics": []}
    if not samples:
        return summary

    algorithm_names = [
        "hot",
        "baseline_content",
        "baseline_cf",
        "baseline_hybrid",
        "word2vec_content",
        "glove_content",
        "word2vec_cf",
        "glove_cf",
        "semantic_hybrid",
        "ppo_rerank",
    ]

    for algorithm in algorithm_names:
        precisions = []
        recalls = []
        ndcgs = []
        coverage_movies = set()
        diversity_scores = []

        for sample in samples:
            if algorithm == "hot":
                watched_ids = sample["train_preferences"].keys()
                ranked_ids = [movie["id"] for movie in _hot_movies(top_k * 2) if movie["id"] not in watched_ids][:top_k]
            else:
                baseline_content_scores = _baseline_content_scores(sample, movies)
                baseline_cf_scores = _baseline_cf_scores(sample, all_preferences)
                semantic_w2v_content = score_content_semantic(sample["train_preferences"], model_name="word2vec")
                semantic_glove_content = score_content_semantic(sample["train_preferences"], model_name="glove")
                semantic_w2v_cf = score_cf_semantic(
                    sample["user_email"],
                    sample["train_preferences"],
                    model_name="word2vec",
                    all_preferences=all_preferences,
                )
                semantic_glove_cf = score_cf_semantic(
                    sample["user_email"],
                    sample["train_preferences"],
                    model_name="glove",
                    all_preferences=all_preferences,
                )

                if algorithm == "baseline_content":
                    ranked_ids = _rank_ids(baseline_content_scores, top_k)
                elif algorithm == "baseline_cf":
                    ranked_ids = _rank_ids(baseline_cf_scores, top_k)
                elif algorithm == "baseline_hybrid":
                    ranked_ids = _rank_ids(_merge_score_maps([baseline_cf_scores, baseline_content_scores], [0.55, 0.45]), top_k)
                elif algorithm == "word2vec_content":
                    ranked_ids = _rank_ids(semantic_w2v_content, top_k)
                elif algorithm == "glove_content":
                    ranked_ids = _rank_ids(semantic_glove_content, top_k)
                elif algorithm == "word2vec_cf":
                    ranked_ids = _rank_ids(semantic_w2v_cf, top_k)
                elif algorithm == "glove_cf":
                    ranked_ids = _rank_ids(semantic_glove_cf, top_k)
                elif algorithm == "semantic_hybrid":
                    ranked_ids = _rank_ids(
                        _merge_score_maps(
                            [semantic_w2v_content, semantic_glove_content, semantic_w2v_cf, semantic_glove_cf],
                            [0.3, 0.3, 0.2, 0.2],
                        ),
                        top_k,
                    )
                elif algorithm == "ppo_rerank":
                    candidate_ids = _rank_ids(
                        _merge_score_maps(
                            [semantic_w2v_content, semantic_glove_content, semantic_w2v_cf, semantic_glove_cf],
                            [0.3, 0.3, 0.2, 0.2],
                        ),
                        max(top_k * 3, top_k),
                    )
                    ranked_ids = _ppo_rerank_ids(candidate_ids, sample["train_preferences"], feature_map, top_k)
                else:
                    ranked_ids = []

            coverage_movies.update(ranked_ids)
            precisions.append(_precision_at_k(ranked_ids, sample["target_movie_id"], top_k))
            recalls.append(_recall_at_k(ranked_ids, sample["target_movie_id"], top_k))
            ndcgs.append(_ndcg_at_k(ranked_ids, sample["target_movie_id"], top_k))
            diversity_scores.append(_list_diversity(ranked_ids, feature_map))

        summary["metrics"].append(
            {
                "algorithm": algorithm,
                "precision_at_k": round(float(np.mean(precisions)), 4),
                "recall_at_k": round(float(np.mean(recalls)), 4),
                "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
                "coverage": round(len(coverage_movies) / max(len(movies), 1), 4),
                "diversity": round(float(np.mean(diversity_scores)), 4),
            }
        )

    summary["metrics"].sort(key=lambda item: item["ndcg_at_k"], reverse=True)
    return summary
