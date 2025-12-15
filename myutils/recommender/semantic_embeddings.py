from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from myutils.query import get_movie_data
from myutils.rl.cache import remember
from services.behavior_service import BehaviorService


EMBEDDING_DIM = 24
WORD2VEC_CACHE_KEY = "semantic:text:word2vec:v1"
GLOVE_CACHE_KEY = "semantic:text:glove:v1"
WORD2VEC_CF_CACHE_KEY = "semantic:cf:word2vec:v1"
GLOVE_CF_CACHE_KEY = "semantic:cf:glove:v1"
SYNONYM_MAP = {
    "悬疑": ["惊悚", "推理"],
    "惊悚": ["悬疑", "紧张"],
    "科幻": ["未来", "宇宙", "设定"],
    "犯罪": ["黑帮", "人性"],
    "剧情": ["叙事", "人物"],
    "亲情": ["治愈", "情感"],
    "dream": ["mind_bending", "sci_fi"],
    "sci-fi": ["future", "space"],
    "thriller": ["suspense", "mystery"],
    "mystery": ["thriller", "suspense"],
}
behavior_service = BehaviorService()


def _normalize_token(token: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5_+-]+", "", token.strip().lower())


def _tokenize_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    raw_tokens = re.split(r"[\s,，。/；;：:()（）|]+", str(text))
    tokens = []
    for token in raw_tokens:
        normalized = _normalize_token(token)
        if normalized:
            tokens.append(normalized)
    return tokens


def _movie_text_tokens(movie: Dict) -> List[str]:
    tokens = []
    tokens.extend(_tokenize_text(movie.get("title")))
    tokens.extend([_normalize_token(token) for token in movie.get("types_list") or movie.get("genres") or [] if token])
    tokens.extend([_normalize_token(token) for token in movie.get("directors_list") or [] if token])
    tokens.extend([_normalize_token(token) for token in movie.get("country_list") or [] if token])
    tokens.extend(_tokenize_text((movie.get("summary") or movie.get("description") or ""))[:24])
    expanded = []
    for token in tokens:
        if not token:
            continue
        expanded.append(token)
        expanded.extend(SYNONYM_MAP.get(token, []))
    return expanded


def _movie_records() -> List[Dict]:
    return get_movie_data() or []


def _movie_map() -> Dict[int, Dict]:
    return {int(movie["id"]): movie for movie in _movie_records()}


def _load_user_preferences() -> Dict[str, Dict[int, float]]:
    preferences: Dict[str, Dict[int, float]] = defaultdict(dict)
    for row in behavior_service.get_rating_events():
        preferences[str(row["user_email"])][int(row["movie_id"])] = float(row["score"] or 0.0)
    return preferences


def _load_user_interaction_sequences() -> List[List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for row in behavior_service.get_interaction_events():
        is_positive = (int(row["behavior_type"]) == 1 and float(row["score"] or 0.0) >= 7.0) or int(row["behavior_type"]) in {2, 3}
        if is_positive:
            grouped[str(row["user_email"])].append(f"movie_{int(row['movie_id'])}")
    return [sequence for sequence in grouped.values() if len(sequence) >= 2]


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 8.0), -8.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _train_word2vec_like(sequences: Sequence[Sequence[str]], dim: int = EMBEDDING_DIM, window: int = 2, epochs: int = 16) -> Dict[str, np.ndarray]:
    vocab = sorted({token for sequence in sequences for token in sequence if token})
    if not vocab:
        return {}
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    rng = np.random.default_rng(42)
    vocab_size = len(vocab)
    input_embeddings = rng.normal(0, 0.08, size=(vocab_size, dim))
    output_embeddings = rng.normal(0, 0.08, size=(vocab_size, dim))

    frequencies = Counter(token for sequence in sequences for token in sequence if token in token_to_idx)
    sampling = np.array([frequencies[idx_to_token[idx]] ** 0.75 for idx in range(vocab_size)], dtype=float)
    sampling = sampling / max(sampling.sum(), 1e-8)

    for _ in range(max(epochs, 1)):
        for sequence in sequences:
            indexed = [token_to_idx[token] for token in sequence if token in token_to_idx]
            for center_pos, center_idx in enumerate(indexed):
                start = max(0, center_pos - window)
                end = min(len(indexed), center_pos + window + 1)
                for context_pos in range(start, end):
                    if context_pos == center_pos:
                        continue
                    context_idx = indexed[context_pos]
                    center_vec = input_embeddings[center_idx]
                    context_vec = output_embeddings[context_idx]
                    positive_score = _sigmoid(float(center_vec @ context_vec))
                    grad = 0.03 * (1.0 - positive_score)
                    input_embeddings[center_idx] += grad * context_vec
                    output_embeddings[context_idx] += grad * center_vec

                    negatives = rng.choice(vocab_size, size=min(4, vocab_size), p=sampling, replace=False)
                    for negative_idx in negatives:
                        if negative_idx == context_idx:
                            continue
                        negative_vec = output_embeddings[negative_idx]
                        negative_score = _sigmoid(float(input_embeddings[center_idx] @ negative_vec))
                        negative_grad = 0.03 * (0.0 - negative_score)
                        input_embeddings[center_idx] += negative_grad * negative_vec
                        output_embeddings[negative_idx] += negative_grad * input_embeddings[center_idx]

    combined = (input_embeddings + output_embeddings) / 2.0
    return {token: combined[idx] for token, idx in token_to_idx.items()}


def _train_glove_like(sequences: Sequence[Sequence[str]], dim: int = EMBEDDING_DIM, window: int = 3, epochs: int = 22) -> Dict[str, np.ndarray]:
    vocab = sorted({token for sequence in sequences for token in sequence if token})
    if not vocab:
        return {}
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    co_occurrence: Dict[Tuple[int, int], float] = defaultdict(float)
    for sequence in sequences:
        indexed = [token_to_idx[token] for token in sequence if token in token_to_idx]
        for center_pos, center_idx in enumerate(indexed):
            start = max(0, center_pos - window)
            end = min(len(indexed), center_pos + window + 1)
            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue
                context_idx = indexed[context_pos]
                distance = abs(center_pos - context_pos)
                co_occurrence[(center_idx, context_idx)] += 1.0 / max(distance, 1)

    rng = np.random.default_rng(123)
    vocab_size = len(vocab)
    word_vectors = rng.normal(0, 0.05, size=(vocab_size, dim))
    context_vectors = rng.normal(0, 0.05, size=(vocab_size, dim))
    word_bias = np.zeros(vocab_size, dtype=float)
    context_bias = np.zeros(vocab_size, dtype=float)
    word_accum = np.ones((vocab_size, dim), dtype=float)
    context_accum = np.ones((vocab_size, dim), dtype=float)
    bias_accum = np.ones(vocab_size, dtype=float)

    x_max = 10.0
    alpha = 0.75
    learning_rate = 0.04
    pairs = list(co_occurrence.items())
    for _ in range(max(epochs, 1)):
        for (left_idx, right_idx), value in pairs:
            weight = (value / x_max) ** alpha if value < x_max else 1.0
            inner = float(word_vectors[left_idx] @ context_vectors[right_idx] + word_bias[left_idx] + context_bias[right_idx] - math.log(max(value, 1e-8)))
            grad_common = weight * inner
            left_grad = grad_common * context_vectors[right_idx]
            right_grad = grad_common * word_vectors[left_idx]

            word_vectors[left_idx] -= (learning_rate / np.sqrt(word_accum[left_idx])) * left_grad
            context_vectors[right_idx] -= (learning_rate / np.sqrt(context_accum[right_idx])) * right_grad
            word_accum[left_idx] += left_grad ** 2
            context_accum[right_idx] += right_grad ** 2

            word_bias[left_idx] -= learning_rate * grad_common / math.sqrt(bias_accum[left_idx])
            context_bias[right_idx] -= learning_rate * grad_common / math.sqrt(bias_accum[right_idx])
            bias_accum[left_idx] += grad_common ** 2
            bias_accum[right_idx] += grad_common ** 2

    combined = (word_vectors + context_vectors) / 2.0
    return {token: combined[idx] for token, idx in token_to_idx.items()}


def _to_serializable_map(vector_map: Dict[int, np.ndarray]) -> Dict[str, List[float]]:
    return {str(item_id): vector.tolist() for item_id, vector in vector_map.items()}


def _text_embedding_payload(model_name: str) -> Dict[str, List[float]]:
    movies = _movie_records()
    sequences = [_movie_text_tokens(movie) for movie in movies]
    if model_name == "glove":
        token_vectors = _train_glove_like(sequences)
    else:
        token_vectors = _train_word2vec_like(sequences)
    movie_vectors: Dict[int, np.ndarray] = {}
    for movie, tokens in zip(movies, sequences):
        vectors = [token_vectors[token] for token in tokens if token in token_vectors]
        if vectors:
            movie_vectors[int(movie["id"])] = np.mean(vectors, axis=0)
    return _to_serializable_map(movie_vectors)


def _interaction_embedding_payload(model_name: str) -> Dict[str, List[float]]:
    sequences = _load_user_interaction_sequences()
    if model_name == "glove":
        token_vectors = _train_glove_like(sequences)
    else:
        token_vectors = _train_word2vec_like(sequences)
    movie_vectors: Dict[int, np.ndarray] = {}
    for token, vector in token_vectors.items():
        if not token.startswith("movie_"):
            continue
        movie_vectors[int(token.split("_", 1)[1])] = vector
    return _to_serializable_map(movie_vectors)


def get_text_embedding_map(model_name: str = "word2vec", force_refresh: bool = False) -> Dict[int, np.ndarray]:
    cache_key = GLOVE_CACHE_KEY if model_name == "glove" else WORD2VEC_CACHE_KEY
    payload = remember(cache_key, lambda: _text_embedding_payload(model_name), ttl=3600, force_refresh=force_refresh)
    return {int(movie_id): np.array(vector, dtype=float) for movie_id, vector in payload.items()}


def get_interaction_embedding_map(model_name: str = "word2vec", force_refresh: bool = False) -> Dict[int, np.ndarray]:
    cache_key = GLOVE_CF_CACHE_KEY if model_name == "glove" else WORD2VEC_CF_CACHE_KEY
    payload = remember(cache_key, lambda: _interaction_embedding_payload(model_name), ttl=3600, force_refresh=force_refresh)
    return {int(movie_id): np.array(vector, dtype=float) for movie_id, vector in payload.items()}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left) or 1.0
    right_norm = np.linalg.norm(right) or 1.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _weighted_profile(preferences: Dict[int, float], embedding_map: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    vectors = []
    weights = []
    for movie_id, score in preferences.items():
        vector = embedding_map.get(int(movie_id))
        if vector is None:
            continue
        vectors.append(vector)
        weights.append(max(float(score) / 10.0, 0.1))
    if not vectors:
        return None
    return np.average(np.vstack(vectors), axis=0, weights=np.array(weights, dtype=float))


def score_content_semantic(preferences: Dict[int, float], model_name: str = "word2vec") -> Dict[int, float]:
    embedding_map = get_text_embedding_map(model_name)
    profile = _weighted_profile(preferences, embedding_map)
    if profile is None:
        return {}
    scores = {}
    watched = set(preferences.keys())
    for movie_id, vector in embedding_map.items():
        if movie_id in watched:
            continue
        scores[movie_id] = _cosine(profile, vector)
    return scores


def score_cf_semantic(
    user_email: str,
    preferences: Dict[int, float],
    model_name: str = "word2vec",
    all_preferences: Optional[Dict[str, Dict[int, float]]] = None,
) -> Dict[int, float]:
    embedding_map = get_interaction_embedding_map(model_name)
    profile = _weighted_profile(preferences, embedding_map)
    if profile is None:
        return {}
    scores = {}
    watched = set(preferences.keys())
    popularity_bias = defaultdict(float)
    if all_preferences:
        for other_user, other_preferences in all_preferences.items():
            if other_user == user_email:
                continue
            for movie_id, score in other_preferences.items():
                if score >= 8 and movie_id not in watched:
                    popularity_bias[movie_id] += score / 10.0

    for movie_id, vector in embedding_map.items():
        if movie_id in watched:
            continue
        scores[movie_id] = _cosine(profile, vector) + 0.03 * popularity_bias.get(movie_id, 0.0)
    return scores


def _scores_to_movies(scores: Dict[int, float], reason: str, top_n: int) -> List[Dict]:
    movies = _movie_map()
    ranked = []
    for movie_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        movie = movies.get(movie_id)
        if not movie:
            continue
        ranked.append(
            {
                "id": movie_id,
                "title": movie["title"],
                "rate": float(movie.get("rating") or movie.get("rate") or 0),
                "cover_url": movie.get("cover_url"),
                "reason": f"{reason}（得分 {score:.3f}）",
                "score": round(float(score), 4),
            }
        )
    return ranked


def word2vec_content_recommend(user_email: str, top_n: int = 10) -> List[Dict]:
    preferences = _load_user_preferences().get(user_email, {})
    if not preferences:
        return []
    return _scores_to_movies(score_content_semantic(preferences, model_name="word2vec"), "Word2Vec 语义内容推荐", top_n)


def glove_content_recommend(user_email: str, top_n: int = 10) -> List[Dict]:
    preferences = _load_user_preferences().get(user_email, {})
    if not preferences:
        return []
    return _scores_to_movies(score_content_semantic(preferences, model_name="glove"), "GloVe 语义内容推荐", top_n)


def word2vec_cf_recommend(user_email: str, top_n: int = 10) -> List[Dict]:
    all_preferences = _load_user_preferences()
    preferences = all_preferences.get(user_email, {})
    if not preferences:
        return []
    return _scores_to_movies(
        score_cf_semantic(user_email, preferences, model_name="word2vec", all_preferences=all_preferences),
        "Word2Vec 交互嵌入协同推荐",
        top_n,
    )


def glove_cf_recommend(user_email: str, top_n: int = 10) -> List[Dict]:
    all_preferences = _load_user_preferences()
    preferences = all_preferences.get(user_email, {})
    if not preferences:
        return []
    return _scores_to_movies(
        score_cf_semantic(user_email, preferences, model_name="glove", all_preferences=all_preferences),
        "GloVe 交互嵌入协同推荐",
        top_n,
    )


def semantic_hybrid_recommend(user_email: Optional[str], top_n: int = 10) -> List[Dict]:
    if not user_email:
        return []
    all_preferences = _load_user_preferences()
    preferences = all_preferences.get(user_email, {})
    if not preferences:
        return []
    score_maps = [
        score_content_semantic(preferences, model_name="word2vec"),
        score_content_semantic(preferences, model_name="glove"),
        score_cf_semantic(user_email, preferences, model_name="word2vec", all_preferences=all_preferences),
        score_cf_semantic(user_email, preferences, model_name="glove", all_preferences=all_preferences),
    ]
    merged = defaultdict(float)
    for index, score_map in enumerate(score_maps):
        weight = 0.3 if index < 2 else 0.2
        for movie_id, score in score_map.items():
            merged[movie_id] += weight * score
    return _scores_to_movies(merged, "语义嵌入混合推荐", top_n)
