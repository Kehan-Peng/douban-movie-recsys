from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - graceful fallback
    pd = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - graceful fallback
    cosine_similarity = None

from .query import init_db, querys
from .recommender.semantic_embeddings import (
    glove_cf_recommend,
    glove_content_recommend,
    semantic_hybrid_recommend,
    word2vec_cf_recommend,
    word2vec_content_recommend,
)
from .rl.local_ppo import rerank_with_local_ppo
from services.behavior_service import BehaviorService


behavior_service = BehaviorService()


def load_movie_data():
    init_db()
    sql = "select id, title, directors, types, country, rate, comment_len, summary, cover_url from movies"
    data = querys(sql, [], "select")
    columns = [
        "id",
        "title",
        "directors",
        "types",
        "country",
        "rate",
        "comment_len",
        "summary",
        "cover_url",
    ]
    if pd is None:
        movies = []
        for row in data:
            movie = dict(zip(columns, row))
            movie["types"] = _split_multi_value(movie.get("types"))
            movie["directors"] = _split_multi_value(movie.get("directors"))
            movie["country"] = _split_multi_value(movie.get("country"))
            movie["rate"] = float(movie.get("rate") or 0)
            movie["comment_len"] = int(movie.get("comment_len") or 0)
            movies.append(movie)
        return movies

    df = pd.DataFrame(data, columns=columns)
    if df.empty:
        return df
    df["types"] = df["types"].apply(_split_multi_value)
    df["directors"] = df["directors"].apply(_split_multi_value)
    df["country"] = df["country"].apply(_split_multi_value)
    df["rate"] = df["rate"].fillna(0).astype(float)
    df["comment_len"] = df["comment_len"].fillna(0).astype(int)
    return df


def load_user_behavior_data():
    columns = ["user_email", "movie_id", "score"]
    data = behavior_service.get_rating_events()
    if pd is None:
        behaviors = [{key: row[key] for key in columns} for row in data]
        rating_matrix: Dict[str, Dict[int, float]] = defaultdict(dict)
        for item in behaviors:
            rating_matrix[item["user_email"]][int(item["movie_id"])] = float(item["score"])
        return behaviors, rating_matrix

    df = pd.DataFrame([{key: row[key] for key in columns} for row in data], columns=columns)
    if df.empty:
        return df, pd.DataFrame()
    df["score"] = df["score"].astype(float)
    rating_matrix = df.pivot_table(index="user_email", columns="movie_id", values="score")
    return df, rating_matrix


def _split_multi_value(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).replace(",", "/").split("/") if item.strip()]


def _extract_movie_id(movie, default: Optional[int] = None) -> int:
    if default is not None:
        return int(default)

    if isinstance(movie, dict):
        if "id" in movie:
            return int(movie["id"])
        if "movie_id" in movie:
            return int(movie["movie_id"])

    try:
        if "id" in movie:
            return int(movie["id"])
    except TypeError:
        pass

    try:
        if "movie_id" in movie:
            return int(movie["movie_id"])
    except TypeError:
        pass

    if hasattr(movie, "name") and movie.name is not None:
        return int(movie.name)

    raise KeyError("id")


def _movie_dict(movie, movie_id: Optional[int] = None) -> Dict:
    current_movie_id = _extract_movie_id(movie, movie_id)

    if isinstance(movie, dict):
        return {
            "id": current_movie_id,
            "title": movie["title"],
            "rate": float(movie["rate"]),
            "cover_url": movie.get("cover_url"),
        }

    return {
        "id": current_movie_id,
        "title": movie["title"],
        "rate": float(movie["rate"]),
        "cover_url": movie.get("cover_url"),
    }


def _dedupe_movie_list(movies: Sequence[Dict], top_n: Optional[int] = None) -> List[Dict]:
    deduped = []
    seen_ids = set()
    for movie in movies:
        movie_id = movie.get("id")
        if movie_id is None or movie_id in seen_ids:
            continue
        seen_ids.add(movie_id)
        deduped.append(movie)
        if top_n is not None and len(deduped) >= top_n:
            break
    return deduped


def _hot_movies(top_n: int = 10) -> List[Dict]:
    sql = """
        select id, title, rate, cover_url
        from movies
        order by rate desc, comment_len desc, title asc
        limit %s
    """
    hot_movies = querys(sql, [top_n], "select")
    return [
        {
            "id": row[0],
            "title": row[1],
            "rate": float(row[2] or 0),
            "cover_url": row[3],
            "reason": "热门高分电影",
        }
        for row in hot_movies
    ]


def _tokenize_movie(movie: Dict) -> List[str]:
    tokens = []
    for field in ("types", "directors", "country"):
        tokens.extend([item.lower() for item in movie.get(field, [])])
    summary = (movie.get("summary") or "").replace("，", " ").replace("。", " ")
    tokens.extend([token.lower() for token in summary.split()[:12]])
    return tokens


def _build_similarity_scores(movie_records: List[Dict], source_movie_ids: Sequence[int]) -> Dict[int, float]:
    token_map = {movie["id"]: set(_tokenize_movie(movie)) for movie in movie_records}
    scores = defaultdict(float)
    for target_movie_id, target_tokens in token_map.items():
        if target_movie_id in source_movie_ids or not target_tokens:
            continue
        total_score = 0.0
        for source_movie_id in source_movie_ids:
            source_tokens = token_map.get(source_movie_id, set())
            if not source_tokens:
                continue
            overlap = len(target_tokens & source_tokens)
            union = len(target_tokens | source_tokens)
            total_score += overlap / union if union else 0.0
        if total_score > 0:
            scores[target_movie_id] = total_score / max(len(source_movie_ids), 1)
    return scores


def recommend_similar_movies(movie_id: int, top_n: int = 6) -> List[Dict]:
    movie_df = load_movie_data()
    if pd is not None and cosine_similarity is not None and not getattr(movie_df, "empty", True):
        feature_matrix = _build_feature_matrix(movie_df)
        sim_matrix = cosine_similarity(feature_matrix)
        sim_df = pd.DataFrame(sim_matrix, index=movie_df["id"], columns=movie_df["id"])
        if movie_id not in sim_df.columns:
            return _hot_movies(top_n)
        sim_scores = sim_df[movie_id].drop(movie_id).sort_values(ascending=False).head(top_n)
        movie_map = movie_df.set_index("id")
        result = []
        for target_id, score in sim_scores.items():
            movie = movie_map.loc[target_id]
            item = _movie_dict(movie, target_id)
            item["reason"] = f"与当前电影内容相似（相似度 {score:.2f}）"
            result.append(item)
        return _dedupe_movie_list(result, top_n)

    movie_records = movie_df if isinstance(movie_df, list) else []
    scores = _build_similarity_scores(movie_records, [movie_id])
    movie_map = {movie["id"]: movie for movie in movie_records}
    top_movie_ids = [item[0] for item in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    result = []
    for target_id in top_movie_ids:
        item = _movie_dict(movie_map[target_id])
        item["reason"] = "与当前电影内容相似"
        result.append(item)
    return _dedupe_movie_list(result or _hot_movies(top_n), top_n)


def _build_feature_matrix(movie_df):
    def one_hot_encoding(series):
        all_tags = sorted({tag for tags in series for tag in tags})
        encoding = pd.DataFrame(0, index=series.index, columns=all_tags)
        for idx, tags in series.items():
            for tag in tags:
                encoding.at[idx, tag] = 1
        return encoding

    type_encoding = one_hot_encoding(movie_df["types"])
    dir_encoding = one_hot_encoding(movie_df["directors"])
    country_encoding = one_hot_encoding(movie_df["country"])
    return pd.concat([type_encoding, dir_encoding, country_encoding], axis=1)


def content_based_recommend(user_email: Optional[str], top_n: int = 10) -> List[Dict]:
    movie_df = load_movie_data()
    behavior_df, _ = load_user_behavior_data()

    if user_email is None:
        return _hot_movies(top_n)

    if pd is not None:
        if getattr(behavior_df, "empty", True) or user_email not in behavior_df["user_email"].values:
            return _hot_movies(top_n)
        user_like_movies = behavior_df[
            (behavior_df["user_email"] == user_email) & (behavior_df["score"] >= 8)
        ]["movie_id"].tolist()
        if not user_like_movies:
            return _hot_movies(top_n)

        feature_matrix = _build_feature_matrix(movie_df)
        sim_matrix = cosine_similarity(feature_matrix) if cosine_similarity is not None else feature_matrix.dot(feature_matrix.T)
        sim_df = pd.DataFrame(sim_matrix, index=movie_df["id"], columns=movie_df["id"])
        available_like_movies = [mid for mid in user_like_movies if mid in sim_df.columns]
        if not available_like_movies:
            return _hot_movies(top_n)

        sim_scores = sim_df[available_like_movies].mean(axis=1)
        watched_movie_ids = set(
            behavior_df[behavior_df["user_email"] == user_email]["movie_id"].tolist()
        )
        sim_scores = sim_scores.drop(labels=[mid for mid in watched_movie_ids if mid in sim_scores.index], errors="ignore")
        top_movie_ids = sim_scores.sort_values(ascending=False).head(top_n).index.tolist()

        movie_map = movie_df.set_index("id")
        result = []
        for target_id in top_movie_ids:
            movie = movie_map.loc[target_id]
            item = _movie_dict(movie, target_id)
            item["reason"] = "与你高分电影的类型/导演/地区相似"
            result.append(item)
        return _dedupe_movie_list(result or _hot_movies(top_n), top_n)

    behavior_records = behavior_df if isinstance(behavior_df, list) else []
    user_like_movies = [
        item["movie_id"]
        for item in behavior_records
        if item["user_email"] == user_email and float(item["score"] or 0) >= 8
    ]
    if not user_like_movies:
        return _hot_movies(top_n)

    watched_movie_ids = {
        item["movie_id"] for item in behavior_records if item["user_email"] == user_email
    }
    movie_records = movie_df if isinstance(movie_df, list) else []
    sim_scores = _build_similarity_scores(movie_records, user_like_movies)
    top_movie_ids = [
        movie_id
        for movie_id, _ in sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
        if movie_id not in watched_movie_ids
    ][:top_n]
    movie_map = {movie["id"]: movie for movie in movie_records}
    result = []
    for target_id in top_movie_ids:
        item = _movie_dict(movie_map[target_id])
        item["reason"] = "与你高分电影的属性相似"
        result.append(item)
    return _dedupe_movie_list(result or _hot_movies(top_n), top_n)


def user_cf_recommend(user_email: str, top_n: int = 10, top_k: int = 20) -> List[Dict]:
    behavior_df, rating_matrix = load_user_behavior_data()
    movie_df = load_movie_data()

    if pd is not None:
        if getattr(rating_matrix, "empty", True) or user_email not in rating_matrix.index:
            return []

        user_similarity = rating_matrix.T.corr(method="pearson").fillna(0)
        similar_users = (
            user_similarity[user_email]
            .drop(labels=[user_email], errors="ignore")
            .sort_values(ascending=False)
        )
        similar_users = similar_users[similar_users > 0].head(top_k)
        if similar_users.empty:
            return []

        user_watched = set(rating_matrix.loc[user_email].dropna().index.tolist())
        movie_scores: Dict[int, float] = defaultdict(float)
        for sim_user, sim_score in similar_users.items():
            sim_user_ratings = rating_matrix.loc[sim_user].dropna()
            for movie_id, score in sim_user_ratings.items():
                if movie_id not in user_watched and score >= 8:
                    movie_scores[int(movie_id)] += float(sim_score) * float(score)

        top_movie_ids = [item[0] for item in sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        if not top_movie_ids:
            return []

        movie_map = movie_df.set_index("id")
        result = []
        for target_id in top_movie_ids:
            if target_id not in movie_map.index:
                continue
            movie = movie_map.loc[target_id]
            item = _movie_dict(movie, target_id)
            item["reason"] = "相似用户高分喜欢这部电影"
            result.append(item)
        return _dedupe_movie_list(result, top_n)

    rating_map = rating_matrix if isinstance(rating_matrix, dict) else {}
    if user_email not in rating_map:
        return []

    target_ratings = rating_map[user_email]
    similarities = {}
    for other_user, other_ratings in rating_map.items():
        if other_user == user_email:
            continue
        sim = _pearson_similarity(target_ratings, other_ratings)
        if sim > 0:
            similarities[other_user] = sim
    similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    if not similar_users:
        return []

    movie_scores: Dict[int, float] = defaultdict(float)
    watched_movie_ids = set(target_ratings.keys())
    for other_user, sim in similar_users:
        for movie_id, score in rating_map[other_user].items():
            if movie_id not in watched_movie_ids and score >= 8:
                movie_scores[movie_id] += sim * score

    movie_records = movie_df if isinstance(movie_df, list) else []
    movie_map = {movie["id"]: movie for movie in movie_records}
    top_movie_ids = [item[0] for item in sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    result = []
    for target_id in top_movie_ids:
        if target_id not in movie_map:
            continue
        item = _movie_dict(movie_map[target_id])
        item["reason"] = "相似用户高分喜欢这部电影"
        result.append(item)
    return _dedupe_movie_list(result, top_n)


def _pearson_similarity(a: Dict[int, float], b: Dict[int, float]) -> float:
    common_items = set(a) & set(b)
    if len(common_items) < 2:
        return 0.0
    a_scores = [a[item] for item in common_items]
    b_scores = [b[item] for item in common_items]
    a_mean = sum(a_scores) / len(a_scores)
    b_mean = sum(b_scores) / len(b_scores)
    numerator = sum((a[item] - a_mean) * (b[item] - b_mean) for item in common_items)
    denominator_left = sqrt(sum((a[item] - a_mean) ** 2 for item in common_items))
    denominator_right = sqrt(sum((b[item] - b_mean) ** 2 for item in common_items))
    denominator = denominator_left * denominator_right
    return numerator / denominator if denominator else 0.0


def _baseline_hybrid_recommend_movies(user_email: Optional[str], top_n: int = 10) -> List[Dict]:
    if not user_email:
        return _hot_movies(top_n)

    cf_result = user_cf_recommend(user_email, top_n)
    if len(cf_result) >= top_n:
        return cf_result[:top_n]

    content_result = content_based_recommend(user_email, top_n)
    merged = _dedupe_movie_list(cf_result + content_result, top_n)
    if len(merged) < top_n:
        merged = _dedupe_movie_list(merged + _hot_movies(top_n), top_n)
    return merged


def _semantic_hybrid_recommend_movies(user_email: Optional[str], top_n: int = 10) -> List[Dict]:
    if not user_email:
        return _hot_movies(top_n)
    semantic_result = semantic_hybrid_recommend(user_email, top_n)
    if len(semantic_result) >= top_n:
        return semantic_result[:top_n]
    fallback = _baseline_hybrid_recommend_movies(user_email, top_n)
    return _dedupe_movie_list(semantic_result + fallback, top_n)


def get_algorithm_recommendations(algorithm: str, user_email: Optional[str], top_n: int = 10) -> List[Dict]:
    algorithm_map = {
        "hot": lambda email, n: _hot_movies(n),
        "baseline_content": lambda email, n: content_based_recommend(email, n),
        "baseline_cf": lambda email, n: user_cf_recommend(email, n) if email else [],
        "baseline_hybrid": lambda email, n: _baseline_hybrid_recommend_movies(email, n),
        "word2vec_content": lambda email, n: word2vec_content_recommend(email, n) if email else [],
        "glove_content": lambda email, n: glove_content_recommend(email, n) if email else [],
        "word2vec_cf": lambda email, n: word2vec_cf_recommend(email, n) if email else [],
        "glove_cf": lambda email, n: glove_cf_recommend(email, n) if email else [],
        "semantic_hybrid": lambda email, n: _semantic_hybrid_recommend_movies(email, n),
    }
    recommender = algorithm_map.get(algorithm)
    if recommender is None:
        raise KeyError(f"unsupported algorithm: {algorithm}")
    return recommender(user_email, top_n)


def recommend_movies(user_email: Optional[str], top_n: int = 10) -> List[Dict]:
    candidate_pool_size = max(top_n * 3, top_n)
    candidate_movies = _semantic_hybrid_recommend_movies(user_email, candidate_pool_size)
    reranked = rerank_with_local_ppo(user_email, candidate_movies, top_n)
    return _dedupe_movie_list(reranked or candidate_movies, top_n)
