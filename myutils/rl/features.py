from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import numpy as np

from datetime import datetime
from .cache import remember, get_cache_backend
from .semantic import get_semantic_tags
from ..query import get_movie_data
from services.behavior_service import BehaviorService

# ====================== 可配置特征参数 ======================
# 强特征：固定字典（类型 + 语义标签）
GENRE_LIST = [
    "action", "comedy", "drama", "sci-fi", "horror", "romance",
    "adventure", "crime", "animation", "documentary",
    "fantasy", "mystery", "thriller", "family", "history"
]
SEMANTIC_TAGS_LIST = [
    "action", "romantic", "comedy", "thriller", "suspense",
    "heartwarming", "sad", "inspiring", "dark", "light",
    "fast-paced", "slow-paced", "violent", "funny", "epic"
]
# 固定字典（全局只生成1次）
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_LIST)}
TAG_TO_IDX = {t: i for i, t in enumerate(SEMANTIC_TAGS_LIST)}

# 弱特征：哈希桶维度（适当提高）
DIRECTOR_DIM = 128          # 原 64 → 128，减少冲突
COUNTRY_DIM = 64            # 原 32 → 64
SUMMARY_DIM = 128           # 原 64 → 128，摘要信息保留更多

# 数值特征维度
MOVIE_NUMERIC_DIM = 4

# 最终电影特征维度
GENRE_DIM = len(GENRE_LIST)
SEMANTIC_DIM = len(SEMANTIC_TAGS_LIST)
MOVIE_FEATURE_DIM = (
    GENRE_DIM + DIRECTOR_DIM + COUNTRY_DIM +
    SUMMARY_DIM + SEMANTIC_DIM + MOVIE_NUMERIC_DIM
)

# 用户统计特征维度（保留）
USER_STATS_DIM = 4
USER_FEATURE_DIM = MOVIE_FEATURE_DIM          # 用户特征与电影特征同维
STATE_VECTOR_DIM = USER_FEATURE_DIM + MOVIE_FEATURE_DIM   # 152 维

# ====================== 缓存版本自动计算（含所有维度变量） ======================
FEATURE_VERSION_HASH = hashlib.md5(
    f"{GENRE_DIM}:{SEMANTIC_DIM}:{DIRECTOR_DIM}:{COUNTRY_DIM}:{SUMMARY_DIM}:{MOVIE_NUMERIC_DIM}".encode()
).hexdigest()[:8]
MOVIE_FEATURE_CACHE_KEY = f"rl:movie-features:{FEATURE_VERSION_HASH}"
USER_HISTORY_CACHE_KEY = f"rl:user-history:{FEATURE_VERSION_HASH}:{{email}}"
USER_FEATURE_CACHE_KEY = f"rl:user-feature:{FEATURE_VERSION_HASH}:{{email}}"
STATE_VECTOR_CACHE_KEY = f"rl:state-vector:{FEATURE_VERSION_HASH}:{{email}}"

behavior_service = BehaviorService()

# 停用词（中英文）
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "for", "with",
    "我", "你", "他", "她", "它", "这", "那", "的", "了", "是", "在", "也", "有", "不", "就"
}

# 负反馈处理策略: "separate" 将正负反馈编码到两个向量通道; "unified" 保持原加权平均
NEGATIVE_FEEDBACK_STRATEGY = "unified"   # 可选 "separate" 或 "unified"

# ====================== 工具函数 ======================
def _hash_bucket(token: str, size: int, namespace: str) -> int:
    """确定性哈希分桶"""
    digest = hashlib.md5(f"{namespace}:{token}".encode()).hexdigest()
    return int(digest, 16) % size

def _multi_hot_fixed(tokens: List[str], mapping: Dict[str, int]) -> np.ndarray:
    """固定词典的 Multi‑Hot，不归一化（保留绝对计数）"""
    vec = np.zeros(len(mapping), dtype=np.float32)
    for t in tokens:
        t = t.strip().lower()
        if t in mapping:
            vec[mapping[t]] += 1.0
    return vec

def _multi_hot_hash(tokens: List[str], size: int, namespace: str, normalize: bool = True) -> np.ndarray:
    """哈希分桶 Multi‑Hot，可选 L1 归一化"""
    vec = np.zeros(size, dtype=np.float32)
    cleaned = [t.strip().lower() for t in tokens if t and t.strip()]
    for t in cleaned:
        vec[_hash_bucket(t, size, namespace)] += 1.0
    if normalize and vec.sum() > 0:
        vec /= vec.sum()
    return vec

def _tokenize_improved(text: Optional[str]) -> List[str]:
    """增强分词：支持简单中英文分割，可选 jieba（需安装）"""
    if not text:
        return []
    text = str(text).lower()

    tokens = re.split(r'[\s,，。/；;：:()（）.!?]+', text)
    return [t for t in tokens if t and len(t) > 1 and t not in STOP_WORDS]

def _normalize_robust(value: Optional[float], v_min: float, v_max: float, default: float = None) -> float:
    """数值归一化，支持缺失值默认"""
    if value is None:
        val = default if default is not None else (v_min + v_max) / 2
    else:
        val = float(value)
    val = np.clip(val, v_min, v_max)
    if v_max == v_min:
        return 0.0
    return (val - v_min) / (v_max - v_min) if v_max > v_min else 0.0

def _time_decay(timestamp: int, half_life_hours: float = 168.0) -> float:
    """指数时间衰减（半衰期默认7天）"""
    if not timestamp or timestamp <= 0:
        return 0.5   # 无时间戳的旧行为给予较低权重
    now = datetime.now().timestamp()
    hours = (now - timestamp) / 3600.0
    decay = np.exp(-hours * np.log(2) / half_life_hours)   # 正确半衰期
    return float(np.clip(decay, 0.05, 1.0))


# ====================== 电影特征（使用预计算缓存） ======================
def _movie_feature_from_record(movie: Dict) -> np.ndarray:
    movie_id = int(movie.get("id", movie.get("movie_id", 0)))
    genres = movie.get("types_list", movie.get("genres", []))
    directors = movie.get("directors_list", _tokenize_improved(movie.get("directors")))
    countries = movie.get("country_list", _tokenize_improved(movie.get("country")))
    summary = _tokenize_improved(movie.get("summary") or movie.get("description"))[:30]
    summary_text = movie.get("summary") or movie.get("description")
    semantic_tags = get_semantic_tags(movie_id, summary_text)

    rating = _normalize_robust(movie.get("rating"), 0, 10)
    comment_len = _normalize_robust(movie.get("comment_len"), 0, 5000)
    year = _normalize_robust(movie.get("release_year"), 1900, 2026)
    duration = _normalize_robust(movie.get("duration"), 20, 300)
    numeric = np.array([rating, comment_len, year, duration], dtype=np.float32)

    return np.concatenate([
        _multi_hot_fixed(genres, GENRE_TO_IDX),
        _multi_hot_hash(directors, DIRECTOR_DIM, "director"),
        _multi_hot_hash(countries, COUNTRY_DIM, "country"),
        _multi_hot_hash(summary, SUMMARY_DIM, "summary"),
        _multi_hot_fixed(semantic_tags, TAG_TO_IDX),
        numeric
    ]).astype(np.float32)

def _load_movie_features_payload() -> Dict[str, List[float]]:
    """批量加载所有电影特征（用于缓存）"""
    movies = get_movie_data() or []
    return {str(m["id"]): _movie_feature_from_record(m).tolist() for m in movies}

def get_movie_feature_map(force_refresh: bool = False) -> Dict[int, np.ndarray]:
    """获取电影特征映射表（ID → 特征向量）"""
    payload = remember(MOVIE_FEATURE_CACHE_KEY, _load_movie_features_payload, ttl=7200, force_refresh=force_refresh)
    return {int(k): np.array(v, dtype=np.float32) for k, v in payload.items()}


# ====================== 用户行为与特征 ======================
def get_user_history(user_email: str, force_refresh: bool = False):
    """获取用户历史行为（带缓存）"""
    return remember(
        USER_HISTORY_CACHE_KEY.format(email=user_email),
        lambda: behavior_service.get_user_history_records(user_email),
        ttl=60, force_refresh=force_refresh
    )

def _build_user_feature(user_email: str) -> List[float]:
    """构建用户特征向量（76维）"""
    history = get_user_history(user_email)
    if not history:
        return np.zeros(USER_FEATURE_DIM, dtype=np.float32).tolist()

    # 预加载电影特征映射（避免重复计算语义标签）
    movie_feat_map = get_movie_feature_map()
    
    # 正负反馈分离（可选）
    if NEGATIVE_FEEDBACK_STRATEGY == "separate":
        pos_weighted = np.zeros(MOVIE_FEATURE_DIM, dtype=np.float32)
        neg_weighted = np.zeros(MOVIE_FEATURE_DIM, dtype=np.float32)
        pos_total_w = 0.0
        neg_total_w = 0.0
    else:
        unified_weighted = np.zeros(MOVIE_FEATURE_DIM, dtype=np.float32)
        unified_total_w = 0.0

    rate_cnt = col_cnt = watch_cnt = 0
    sum_score = 0.0

    for item in history:
        movie_id = item.get("movie_id")
        feat = movie_feat_map.get(movie_id)  # 直接使用预计算特征，若无则跳过
        if feat is None:
            feat = _movie_feature_from_record(item)

        bt = item["behavior_type"]
        ts = item.get("timestamp", 0)
        decay = _time_decay(ts)
        s = float(item.get("score", 0)) if bt == 1 else 0.0

        if bt == 1:
            sum_score += s
            rate_cnt += 1
            # 评分映射：0-10 → -1..1
            w = (s - 5) / 5.0
            w = np.clip(w, -1.0, 1.0) * decay
        elif bt == 2:
            col_cnt += 1
            w = 0.8 * decay
        else:  # 观看（bt=3或其他）
            watch_cnt += 1
            w = 0.4 * decay

        if NEGATIVE_FEEDBACK_STRATEGY == "separate":
            if w > 0:
                pos_weighted += feat * w
                pos_total_w += w
            elif w < 0:
                neg_weighted += feat * (-w)   # 负向绝对值
                neg_total_w += (-w)
        else:
            unified_weighted += feat * w
            unified_total_w += abs(w)        # 绝对值保证分母为正

    if NEGATIVE_FEEDBACK_STRATEGY == "separate":
        pos_pref = pos_weighted / max(pos_total_w, 1e-6) if pos_total_w > 0 else np.zeros(MOVIE_FEATURE_DIM, dtype=np.float32)
        neg_pref = neg_weighted / max(neg_total_w, 1e-6) if neg_total_w > 0 else np.zeros(MOVIE_FEATURE_DIM, dtype=np.float32)
        # 将正负偏好拼接（简单做法：正 - 负），也可保留两个向量但需调整维度
        pref = pos_pref - neg_pref
    else:
        pref = unified_weighted / max(unified_total_w, 1e-6)

    # 用户统计特征（归一化）
    avg_score = sum_score / max(rate_cnt, 1)
    stats = np.array([
        _normalize_robust(avg_score, 0, 10),
        _normalize_robust(rate_cnt, 0, 200),      # 增大上限
        _normalize_robust(col_cnt, 0, 200),
        _normalize_robust(watch_cnt, 0, 500)
    ], dtype=np.float32)

    final = np.concatenate([pref[:MOVIE_FEATURE_DIM - USER_STATS_DIM], stats])
    return final[:USER_FEATURE_DIM].tolist()

def get_user_feature(user_email: str, force_refresh: bool = False) -> np.ndarray:
    """获取用户特征（缓存）"""
    cached = remember(
        USER_FEATURE_CACHE_KEY.format(email=user_email),
        lambda: _build_user_feature(user_email),
        ttl=300, force_refresh=force_refresh   # TTL 从 60s 提升到 300s
    )
    return np.array(cached, dtype=np.float32)


# ====================== PPO 状态向量（支持候选电影） ======================
def _build_state_vector(user_email: str, candidate_movie_id: Optional[int] = None) -> List[float]:
    """
    构建状态向量：
    - 如果提供了 candidate_movie_id，返回 [user_feature, movie_feature] (152维)
    - 否则只返回 user_feature (76维) ，保持向后兼容
    """
    uf = get_user_feature(user_email)
    if candidate_movie_id is not None:
        movie_map = get_movie_feature_map()
        mf = movie_map.get(candidate_movie_id, np.zeros(MOVIE_FEATURE_DIM, dtype=np.float32))
        state = np.concatenate([uf, mf])
    else:
        state = uf
    return state.tolist()

def get_state_vector(user_email: str, candidate_movie_id: Optional[int] = None, force_refresh=False) -> np.ndarray:
    uf = get_user_feature(user_email, force_refresh=force_refresh)
    if candidate_movie_id is not None:
        mf = get_movie_feature_map().get(candidate_movie_id, np.zeros(MOVIE_FEATURE_DIM, dtype=np.float32))
        return np.concatenate([uf, mf])
    return uf


# ====================== 缓存失效 ======================
def invalidate_user_feature_cache(user_email: Optional[str]) -> None:
    """用户相关缓存失效"""
    if not user_email:
        return
    b = get_cache_backend()
    b.delete(USER_HISTORY_CACHE_KEY.format(email=user_email))
    b.delete(USER_FEATURE_CACHE_KEY.format(email=user_email))
    b.delete(STATE_VECTOR_CACHE_KEY.format(email=user_email))

def invalidate_movie_feature_cache() -> None:
    """全局电影特征缓存失效"""
    get_cache_backend().delete(MOVIE_FEATURE_CACHE_KEY)