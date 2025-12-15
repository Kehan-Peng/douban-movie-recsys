from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import numpy as np

from .cache import remember, get_cache_backend
from .semantic import get_semantic_tags
from ..query import get_movie_data
from services.behavior_service import BehaviorService


GENRE_DIM = 10
DIRECTOR_DIM = 8
COUNTRY_DIM = 6
SUMMARY_DIM = 4
SEMANTIC_DIM = 6
MOVIE_NUMERIC_DIM = 4
MOVIE_FEATURE_DIM = GENRE_DIM + DIRECTOR_DIM + COUNTRY_DIM + SUMMARY_DIM + SEMANTIC_DIM + MOVIE_NUMERIC_DIM
USER_STATS_DIM = 4
USER_FEATURE_DIM = MOVIE_FEATURE_DIM
STATE_VECTOR_DIM = USER_FEATURE_DIM + MOVIE_FEATURE_DIM

MOVIE_FEATURE_CACHE_KEY = "rl:movie-features:v1"
USER_HISTORY_CACHE_KEY = "rl:user-history:{email}:v1"
USER_FEATURE_CACHE_KEY = "rl:user-feature:{email}:v1"
STATE_VECTOR_CACHE_KEY = "rl:state-vector:{email}:v1"
behavior_service = BehaviorService()


def _hash_bucket(token: str, size: int, namespace: str) -> int:
    digest = hashlib.md5(f"{namespace}:{token}".encode("utf-8")).hexdigest()
    return int(digest, 16) % size


def _multi_hot(tokens: List[str], size: int, namespace: str) -> np.ndarray:
    vector = np.zeros(size, dtype=float)
    cleaned_tokens = [token.strip().lower() for token in tokens if token and token.strip()]
    if not cleaned_tokens:
        return vector
    for token in cleaned_tokens:
        vector[_hash_bucket(token, size, namespace)] += 1.0
    vector /= max(vector.sum(), 1.0)
    return vector


def _tokenize_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [token for token in re.split(r"[\s,，。/；;：:()（）]+", str(text)) if token]


def _normalize(value: Optional[float], scale: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    current = float(value or 0.0) / scale if scale else 0.0
    return float(min(max(current, min_value), max_value))


def _movie_feature_from_record(movie: Dict) -> np.ndarray:
    movie_id = int(movie.get("id") or movie.get("movie_id") or 0)
    genres = movie.get("types_list") or movie.get("genres") or []
    directors = movie.get("directors_list") or _tokenize_text(movie.get("directors"))
    countries = movie.get("country_list") or _tokenize_text(movie.get("country"))
    summary_tokens = _tokenize_text(movie.get("summary") or movie.get("description"))[:12]
    semantic_tags = get_semantic_tags(movie_id, movie.get("summary") or movie.get("description"))
    numeric = np.array(
        [
            _normalize(movie.get("rating") or movie.get("rate"), 10),
            _normalize(movie.get("comment_len"), 5000),
            _normalize((movie.get("release_date") or movie.get("release_year") or 1950) - 1950, 100),
            _normalize(movie.get("duration"), 240),
        ],
        dtype=float,
    )
    return np.concatenate(
        [
            _multi_hot(genres, GENRE_DIM, "genre"),
            _multi_hot(directors, DIRECTOR_DIM, "director"),
            _multi_hot(countries, COUNTRY_DIM, "country"),
            _multi_hot(summary_tokens, SUMMARY_DIM, "summary"),
            _multi_hot(semantic_tags, SEMANTIC_DIM, "semantic"),
            numeric,
        ]
    )


def _load_movie_features_payload() -> Dict[str, List[float]]:
    movies = get_movie_data() or []
    return {str(movie["id"]): _movie_feature_from_record(movie).tolist() for movie in movies}


def get_movie_feature_map(force_refresh: bool = False) -> Dict[int, np.ndarray]:
    payload = remember(
        MOVIE_FEATURE_CACHE_KEY,
        _load_movie_features_payload,
        ttl=3600,
        force_refresh=force_refresh,
    )
    return {int(movie_id): np.array(values, dtype=float) for movie_id, values in payload.items()}


def _load_user_history(user_email: str) -> List[Dict]:
    return behavior_service.get_user_history_records(user_email)


def get_user_history(user_email: str, force_refresh: bool = False) -> List[Dict]:
    return remember(
        USER_HISTORY_CACHE_KEY.format(email=user_email),
        lambda: _load_user_history(user_email),
        ttl=300,
        force_refresh=force_refresh,
    )


def _build_user_feature(user_email: str) -> List[float]:
    history = get_user_history(user_email)
    if not history:
        cold_start = np.zeros(USER_FEATURE_DIM, dtype=float)
        cold_start[-1] = 1.0
        return cold_start.tolist()

    weighted_sum = np.zeros(MOVIE_FEATURE_DIM, dtype=float)
    high_score_vectors = []
    total_weight = 0.0
    rating_count = 0
    collect_count = 0
    watched_count = 0
    avg_score = 0.0

    for item in history:
        movie_feature = _movie_feature_from_record(item)
        if item["behavior_type"] == 1:
            rating_count += 1
            score = float(item["score"] or 0.0)
            avg_score += score
            weight = max(score / 10.0, 0.1)
            if score >= 8:
                high_score_vectors.append(movie_feature)
        elif item["behavior_type"] == 2:
            collect_count += 1
            weight = 0.7
        else:
            watched_count += 1
            weight = 0.4

        weighted_sum += movie_feature * weight
        total_weight += weight

    preference_vector = weighted_sum / max(total_weight, 1.0)
    stats_vector = np.array(
        [
            _normalize(avg_score / max(rating_count, 1), 10),
            _normalize(rating_count, 20),
            _normalize(collect_count, 20),
            _normalize(watched_count, 20),
        ],
        dtype=float,
    )

    user_vector = np.concatenate([preference_vector[: MOVIE_FEATURE_DIM - USER_STATS_DIM], stats_vector])
    return user_vector.tolist()


def get_user_feature(user_email: str, force_refresh: bool = False) -> np.ndarray:
    payload = remember(
        USER_FEATURE_CACHE_KEY.format(email=user_email),
        lambda: _build_user_feature(user_email),
        ttl=300,
        force_refresh=force_refresh,
    )
    return np.array(payload, dtype=float)


def _build_state_vector(user_email: str) -> List[float]:
    history = get_user_history(user_email)
    movie_features = get_movie_feature_map()
    user_feature = get_user_feature(user_email).tolist()

    high_score_feature_sum = np.zeros(MOVIE_FEATURE_DIM, dtype=float)
    count = 0
    for item in history:
        score = item.get("score")
        if item["behavior_type"] == 1 and score is not None and score >= 8 and item["movie_id"] in movie_features:
            high_score_feature_sum += movie_features[item["movie_id"]]
            count += 1

    if count == 0 and history:
        for item in history[:3]:
            feature = movie_features.get(item["movie_id"])
            if feature is not None:
                high_score_feature_sum += feature
                count += 1

    recent_preference = (
        high_score_feature_sum / max(count, 1)
        if count
        else np.zeros(MOVIE_FEATURE_DIM, dtype=float)
    )
    return np.concatenate([np.array(user_feature, dtype=float), recent_preference]).tolist()


def get_state_vector(user_email: str, force_refresh: bool = False) -> np.ndarray:
    payload = remember(
        STATE_VECTOR_CACHE_KEY.format(email=user_email),
        lambda: _build_state_vector(user_email),
        ttl=300,
        force_refresh=force_refresh,
    )
    return np.array(payload, dtype=float)


def invalidate_user_feature_cache(user_email: Optional[str]) -> None:
    if not user_email:
        return
    backend = get_cache_backend()
    backend.delete(USER_HISTORY_CACHE_KEY.format(email=user_email))
    backend.delete(USER_FEATURE_CACHE_KEY.format(email=user_email))
    backend.delete(STATE_VECTOR_CACHE_KEY.format(email=user_email))


def invalidate_movie_feature_cache() -> None:
    get_cache_backend().delete(MOVIE_FEATURE_CACHE_KEY)
