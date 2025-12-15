from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests


DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
]

BASE_DIR = Path(__file__).resolve().parents[2]
CRAWLER_DIR = BASE_DIR / "runtime" / "crawler"
CHECKPOINT_DIR = CRAWLER_DIR / "checkpoints"
STATUS_FILE = CRAWLER_DIR / "crawler_status.json"


@dataclass
class CrawlConfig:
    min_delay: float = 1.1
    max_delay: float = 2.6
    timeout: int = 15
    retries: int = 3
    user_agents: Optional[List[str]] = None
    proxy_file: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "CrawlConfig":
        proxy_path = os.getenv("DOUBAN_PROXY_FILE")
        return cls(
            min_delay=float(os.getenv("DOUBAN_MIN_DELAY", "1.1")),
            max_delay=float(os.getenv("DOUBAN_MAX_DELAY", "2.6")),
            timeout=int(os.getenv("DOUBAN_TIMEOUT", "15")),
            retries=int(os.getenv("DOUBAN_RETRIES", "3")),
            user_agents=DEFAULT_USER_AGENTS,
            proxy_file=Path(proxy_path) if proxy_path else None,
        )


def _ensure_runtime_dirs() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_proxy_pool(proxy_file: Optional[Path]) -> List[str]:
    if not proxy_file or not proxy_file.exists():
        return []
    return [line.strip() for line in proxy_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_movie_record(record: Dict) -> Dict:
    cleaned = dict(record)
    cleaned["title"] = _normalize_whitespace(cleaned.get("title", ""))
    cleaned["genre"] = "/".join(
        sorted({item.strip() for item in str(cleaned.get("genre", "")).replace(",", "/").split("/") if item.strip()})
    )
    cleaned["summary"] = _normalize_whitespace(cleaned.get("summary", ""))
    cleaned["country"] = "/".join(
        item.strip() for item in str(cleaned.get("country", "")).replace(",", "/").split("/") if item.strip()
    )
    cleaned["directors"] = "/".join(
        item.strip() for item in str(cleaned.get("directors", "")).replace(",", "/").split("/") if item.strip()
    )
    cleaned["duration"] = int(cleaned.get("duration") or 0)
    cleaned["release_year"] = int(cleaned.get("release_year") or 0)
    cleaned["rating"] = float(cleaned.get("rating") or 0)
    cleaned["subject_id"] = str(cleaned.get("subject_id") or "")
    cleaned["comment_len"] = int(cleaned.get("comment_len") or 0)
    return cleaned


def dedupe_records(records: Sequence[Dict], key_fields: Sequence[str]) -> List[Dict]:
    merged = {}
    for item in records:
        key = tuple(str(item.get(field) or "") for field in key_fields)
        if not any(key):
            continue
        merged[key] = item
    return list(merged.values())


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def merge_csv_rows(path: Path, incoming_rows: Sequence[Dict], key_fields: Sequence[str], fieldnames: Sequence[str]) -> List[Dict]:
    existing = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            existing = list(csv.DictReader(csv_file))
    merged = dedupe_records(existing + list(incoming_rows), key_fields)
    write_csv(path, merged, fieldnames)
    return merged


def load_checkpoint(name: str) -> Dict:
    _ensure_runtime_dirs()
    checkpoint_path = CHECKPOINT_DIR / f"{name}.json"
    if not checkpoint_path.exists():
        return {}
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def save_checkpoint(name: str, payload: Dict) -> None:
    _ensure_runtime_dirs()
    checkpoint_path = CHECKPOINT_DIR / f"{name}.json"
    checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_crawler_status(job_name: str, payload: Dict) -> None:
    _ensure_runtime_dirs()
    status = {}
    if STATUS_FILE.exists():
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    status[job_name] = {
        **payload,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


class DoubanHttpClient:
    def __init__(self, config: Optional[CrawlConfig] = None) -> None:
        self.config = config or CrawlConfig.from_env()
        self.session = requests.Session()
        self.proxy_pool = _load_proxy_pool(self.config.proxy_file)

    def _headers(self) -> Dict[str, str]:
        user_agents = self.config.user_agents or DEFAULT_USER_AGENTS
        return {
            "User-Agent": random.choice(user_agents),
            "Referer": "https://movie.douban.com/",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    def _proxies(self) -> Optional[Dict[str, str]]:
        if not self.proxy_pool:
            return None
        proxy = random.choice(self.proxy_pool)
        return {"http": proxy, "https": proxy}

    def get_html(self, url: str) -> str:
        last_error = None
        for _ in range(max(self.config.retries, 1)):
            try:
                response = self.session.get(
                    url,
                    headers=self._headers(),
                    timeout=self.config.timeout,
                    proxies=self._proxies(),
                )
                response.raise_for_status()
                time.sleep(random.uniform(self.config.min_delay, self.config.max_delay))
                return response.text
            except requests.RequestException as exc:  # pragma: no cover - network behavior
                last_error = exc
                time.sleep(random.uniform(self.config.min_delay, self.config.max_delay) + 0.8)
        raise RuntimeError(f"请求失败: {url} ({last_error})")


def parse_top250_page(html: str) -> List[Dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - optional dependency at runtime
        raise RuntimeError("缺少 beautifulsoup4 依赖，请先执行 pip install -r requirements.txt") from exc
    soup = BeautifulSoup(html, "html.parser")
    movies = []
    for item in soup.select("div.item"):
        title_node = item.select_one("span.title")
        rating_node = item.select_one("span.rating_num")
        detail_link = item.select_one("div.hd a")
        info_node = item.select_one("div.bd p")
        summary_node = item.select_one("span.inq")
        info_lines = [
            _normalize_whitespace(line)
            for line in (info_node.get_text("\n", strip=True).split("\n") if info_node else [])
            if _normalize_whitespace(line)
        ]
        info_text = " ".join(info_lines)
        subject_id_match = re.search(r"/subject/(\d+)/", detail_link["href"]) if detail_link else None
        year_match = re.search(r"\b(19|20)\d{2}\b", info_text)
        duration_match = re.search(r"(\d+)\s*分钟", info_text)

        metadata = [item.strip() for item in re.split(r"/", info_text) if item.strip()]
        directors = []
        countries = []
        genres = []
        if metadata:
            directors_match = re.search(r"导演:\s*([^主演]+?)(?:主演:|$)", info_text)
            if directors_match:
                directors = [seg.strip() for seg in re.split(r"\s+", directors_match.group(1)) if seg.strip()]
            countries = [token for token in metadata if re.fullmatch(r"[\u4e00-\u9fa5A-Za-z· ]{2,20}", token)]
            genres = [token for token in metadata if token in info_text and len(token) <= 12]

        movies.append(
            clean_movie_record(
                {
                    "subject_id": subject_id_match.group(1) if subject_id_match else "",
                    "title": title_node.get_text(strip=True) if title_node else "",
                    "rating": rating_node.get_text(strip=True) if rating_node else 0,
                    "genre": "/".join(genres[:4]),
                    "release_year": year_match.group(0) if year_match else 0,
                    "duration": duration_match.group(1) if duration_match else 0,
                    "summary": summary_node.get_text(strip=True) if summary_node else "",
                    "country": "/".join(countries[:3]),
                    "directors": "/".join(directors[:3]),
                    "detail_url": detail_link["href"] if detail_link else "",
                }
            )
        )
    return dedupe_records(movies, ["subject_id", "title"])


def parse_comment_page(html: str, subject_id: str, movie_title: str) -> List[Dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - optional dependency at runtime
        raise RuntimeError("缺少 beautifulsoup4 依赖，请先执行 pip install -r requirements.txt") from exc
    soup = BeautifulSoup(html, "html.parser")
    comments = []
    for item in soup.select("div.comment-item"):
        short_node = item.select_one("span.short")
        user_node = item.select_one("span.comment-info a")
        vote_node = item.select_one("span.votes")
        rating_node = item.select_one("span.comment-info span.rating")
        time_node = item.select_one("span.comment-time")
        if not short_node:
            continue
        rating_value = 0
        if rating_node and rating_node.get("class"):
            for class_name in rating_node.get("class", []):
                match = re.search(r"allstar(\d+)0", class_name)
                if match:
                    rating_value = int(match.group(1)) / 2
                    break
        comments.append(
            {
                "subject_id": subject_id,
                "movie_title": movie_title,
                "comment_user": _normalize_whitespace(user_node.get_text(strip=True) if user_node else "匿名用户"),
                "comment_text": _normalize_whitespace(short_node.get_text(strip=True)),
                "comment_votes": int(vote_node.get_text(strip=True) or 0) if vote_node else 0,
                "comment_rating": rating_value,
                "comment_time": time_node.get("title", "") if time_node else "",
            }
        )
    return dedupe_records(comments, ["subject_id", "comment_user", "comment_text"])
