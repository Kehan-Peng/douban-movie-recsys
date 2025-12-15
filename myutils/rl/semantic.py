from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests

from .cache import remember


SEMANTIC_TAG_CACHE_KEY = "rl:semantic-tags:{movie_id}:v1"
SEMANTIC_BUCKETS = {
    "narrative_style": ["叙事", "剧情", "结构", "悬疑", "烧脑", "反转"],
    "visual_language": ["镜头", "画面", "摄影", "构图", "视觉", "色彩"],
    "emotion": ["亲情", "爱情", "友情", "治愈", "温暖", "悲伤", "感动"],
    "tempo": ["节奏", "紧张", "刺激", "慢热", "激烈"],
    "world_building": ["科幻", "世界观", "宇宙", "未来", "设定", "梦境"],
    "humanity": ["人性", "成长", "自由", "信念", "救赎", "社会"],
}


def _local_semantic_tags(text: Optional[str]) -> List[str]:
    content = str(text or "")
    tags = []
    for tag, keywords in SEMANTIC_BUCKETS.items():
        if any(keyword in content for keyword in keywords):
            tags.append(tag)
    return tags


def _remote_semantic_tags(text: Optional[str]) -> Optional[List[str]]:
    base_url = os.getenv("MOVIE_LLM_BASE_URL")
    api_key = os.getenv("MOVIE_LLM_API_KEY")
    model = os.getenv("MOVIE_LLM_MODEL", "gpt-4o-mini")
    if not base_url or not api_key or not text:
        return None

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是电影语义标注器，只返回最多6个英文短标签，使用逗号分隔。",
                    },
                    {
                        "role": "user",
                        "content": f"请为以下电影简介提取叙事风格、镜头语言、情绪基调相关标签：{text}",
                    },
                ],
                "temperature": 0.2,
            },
            timeout=12,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        tags = [tag.strip().lower().replace(" ", "_") for tag in str(content).split(",") if tag.strip()]
        return tags[:6] or None
    except Exception:  # pragma: no cover - optional remote adapter
        return None


def get_semantic_tags(movie_id: int, summary: Optional[str]) -> List[str]:
    def builder() -> List[str]:
        remote = _remote_semantic_tags(summary)
        return remote or _local_semantic_tags(summary)

    return remember(SEMANTIC_TAG_CACHE_KEY.format(movie_id=movie_id), builder, ttl=24 * 3600)
