from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Optional

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover - optional dependency
    redis_lib = None


DEFAULT_CACHE_TTL = int(os.getenv("MOVIE_CACHE_TTL_SECONDS", "300"))


class InMemoryCache:
    def __init__(self) -> None:
        self._store = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            expires_at, value = entry
            if expires_at and expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> Any:
        expires_at = time.time() + ttl if ttl else None
        with self._lock:
            self._store[key] = (expires_at, value)
        return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


class RedisCache:
    def __init__(self, url: str) -> None:
        self._client = redis_lib.from_url(url, decode_responses=True)

    def get(self, key: str) -> Any:
        value = self._client.get(key)
        if value is None:
            return None
        return json.loads(value)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> Any:
        payload = json.dumps(value, ensure_ascii=False)
        if ttl:
            self._client.setex(key, ttl, payload)
        else:
            self._client.set(key, payload)
        return value

    def delete(self, key: str) -> None:
        self._client.delete(key)


_CACHE_BACKEND = None


def get_cache_backend():
    global _CACHE_BACKEND
    if _CACHE_BACKEND is not None:
        return _CACHE_BACKEND

    redis_url = os.getenv("MOVIE_REDIS_URL")
    if redis_url and redis_lib is not None:
        try:
            backend = RedisCache(redis_url)
            backend._client.ping()
            _CACHE_BACKEND = backend
            return _CACHE_BACKEND
        except Exception:
            pass

    _CACHE_BACKEND = InMemoryCache()
    return _CACHE_BACKEND


def remember(
    key: str,
    builder: Callable[[], Any],
    ttl: Optional[int] = DEFAULT_CACHE_TTL,
    force_refresh: bool = False,
) -> Any:
    backend = get_cache_backend()
    if not force_refresh:
        cached = backend.get(key)
        if cached is not None:
            return cached
    value = builder()
    backend.set(key, value, ttl=ttl)
    return value
