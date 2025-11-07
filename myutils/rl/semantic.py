from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional
from rl.cache import remember
from rl.llm_client import llm_client


SEMANTIC_TAG_CACHE_KEY = "rl:semantic-tags:{movie_id}:v1"

# 【全局标准化：中文关键词 → 细粒度英文标签】
SEMANTIC_KEYWORD_MAP = {
    "叙事": ["linear_story", "story_driven"],
    "剧情": ["drama", "character_story"],
    "结构": ["complex_plot", "well_structured"],
    "悬疑": ["suspense", "mystery"],
    "烧脑": ["mind_bending", "intellectual"],
    "反转": ["plot_twist", "unexpected"],

    "镜头": ["cinematic", "great_shots"],
    "画面": ["visual_beautiful", "stunning_visuals"],
    "摄影": ["excellent_cinematography"],
    "构图": ["well_composed"],
    "视觉": ["visual_style", "striking_visuals"],
    "色彩": ["vibrant_colors", "color_graded"],

    "亲情": ["family_bond", "family_love"],
    "爱情": ["romance", "love_story"],
    "友情": ["friendship", "brotherhood"],
    "治愈": ["healing", "warm_feelings"],
    "温暖": ["warm", "heartwarming"],
    "悲伤": ["sad", "emotional_drama"],
    "感动": ["touching", "emotional"],

    "节奏": ["well_paced"],
    "紧张": ["tense", "thrilling"],
    "刺激": ["exciting", "action_packed"],
    "慢热": ["slow_burn"],
    "激烈": ["intense", "fast_paced"],

    "科幻": ["sci_fi", "science_fiction"],
    "世界观": ["world_building", "rich_lore"],
    "宇宙": ["space", "cosmic"],
    "未来": ["futuristic", "dystopian"],
    "设定": ["unique_setting", "high_concept"],
    "梦境": ["dreamlike", "surreal"],

    "人性": ["human_nature", "philosophical"],
    "成长": ["coming_of_age", "character_growth"],
    "自由": ["freedom", "liberation"],
    "信念": ["faith", "hope"],
    "救赎": ["redemption", "forgiveness"],
    "社会": ["social_commentary", "critical_realism"],
}


def _local_semantic_tags(text: Optional[str]) -> List[str]:
    content = str(text or "")
    tags = set()

    # 正确写法：匹配中文关键词 → 取出英文标签
    for cn_key, en_tags in SEMANTIC_KEYWORD_MAP.items():
        if cn_key in content:
            tags.update(en_tags)

    return list(tags)[:6]  # 最多6个，去重


def _remote_semantic_tags(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None

    try:
        prompt = f"""
任务：给电影简介生成英文语义标签。
规则：
1. 只输出英文小写标签，多个单词用下划线 _ 连接
2. 单个标签内最多2个下划线
3. 绝对不要输出中文
4. 不要输出任何说明文字
5. 最多输出6个，逗号分隔
6. 不要解释，不要多余内容，只输出标签

示例：sci_fi, space_exploration, space, adventure

电影简介：{text}
"""
        result = llm_client.generate(prompt, temperature=0.2, max_tokens=256)

        if result.startswith(("HTTP错误", "API调用失败")):
            return None

        tags = [
            t.strip().lower().replace(" ", "_")
            for t in result.split(",")
            if t.strip() and t.isascii()
        ]
        return tags[:6]

    except Exception:
        return None


def get_semantic_tags(movie_id: int, summary: Optional[str]) -> List[str]:
    def builder():
        remote = _remote_semantic_tags(summary)
        if remote:
            return remote
        return _local_semantic_tags(summary)

    return remember(
        SEMANTIC_TAG_CACHE_KEY.format(movie_id=movie_id),
        builder,
        ttl=24 * 3600
    )


if __name__ == "__main__":
    test_summary = "一部科幻烧脑电影，讲述人类穿越星际寻找新家园。"
    print("标签：", get_semantic_tags(123, test_summary))