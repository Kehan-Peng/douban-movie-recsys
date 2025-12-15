from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

from ..query import generate_password_hash, init_db, querys
from services.behavior_service import BehaviorService
from .core import (
    BASE_DIR,
    DoubanHttpClient,
    clean_movie_record,
    dedupe_records,
    load_checkpoint,
    merge_csv_rows,
    parse_comment_page,
    parse_top250_page,
    save_checkpoint,
    update_crawler_status,
    write_csv,
)


MOVIE_FIELDS = [
    "subject_id",
    "title",
    "rating",
    "genre",
    "release_year",
    "duration",
    "summary",
    "country",
    "directors",
    "detail_url",
    "comment_len",
]
COMMENT_FIELDS = [
    "subject_id",
    "movie_title",
    "comment_user",
    "comment_text",
    "comment_votes",
    "comment_rating",
    "comment_time",
]
BEHAVIOR_FIELDS = ["user_email", "movie_title", "movie_subject_id", "behavior_type", "score", "create_time"]
behavior_service = BehaviorService()


def crawl_top_movies(
    pages: int = 8,
    output_csv: Optional[Path] = None,
    resume: bool = True,
    checkpoint_name: str = "movies",
) -> Dict:
    output_csv = output_csv or (BASE_DIR / "datas.csv")
    client = DoubanHttpClient()
    checkpoint = load_checkpoint(checkpoint_name) if resume else {}
    start_page = int(checkpoint.get("next_page", 0))
    all_rows: List[Dict] = []

    for page in range(start_page, pages):
        url = f"https://movie.douban.com/top250?start={page * 25}"
        html = client.get_html(url)
        page_rows = parse_top250_page(html)
        all_rows.extend(page_rows)
        merged = merge_csv_rows(output_csv, all_rows, ["subject_id", "title"], MOVIE_FIELDS)
        save_checkpoint(checkpoint_name, {"next_page": page + 1, "rows": len(merged), "output_csv": str(output_csv)})
        update_crawler_status(
            "crawl_movies",
            {"status": "running", "next_page": page + 1, "movie_count": len(merged), "output_csv": str(output_csv)},
        )
        all_rows = []

    merged_rows = []
    with output_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        merged_rows = list(csv.DictReader(csv_file))
    update_crawler_status(
        "crawl_movies",
        {"status": "completed", "movie_count": len(merged_rows), "pages": pages, "output_csv": str(output_csv)},
    )
    return {"output_csv": str(output_csv), "movie_count": len(merged_rows), "pages": pages}


def crawl_movie_comments(
    movie_csv: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    pages_per_movie: int = 3,
    limit_movies: Optional[int] = None,
    resume: bool = True,
    checkpoint_name: str = "comments",
) -> Dict:
    movie_csv = movie_csv or (BASE_DIR / "datas.csv")
    output_csv = output_csv or (BASE_DIR / "comments_dataset.csv")
    if not movie_csv.exists():
        raise FileNotFoundError(f"未找到电影数据文件: {movie_csv}")

    with movie_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        movies = [clean_movie_record(row) for row in csv.DictReader(csv_file)]
    if limit_movies:
        movies = movies[:limit_movies]

    client = DoubanHttpClient()
    checkpoint = load_checkpoint(checkpoint_name) if resume else {}
    start_index = int(checkpoint.get("next_movie_index", 0))
    pending_rows: List[Dict] = []

    for movie_index in range(start_index, len(movies)):
        movie = movies[movie_index]
        subject_id = movie.get("subject_id")
        if not subject_id:
            continue
        comment_count = 0
        for page in range(pages_per_movie):
            url = (
                f"https://movie.douban.com/subject/{subject_id}/comments"
                f"?start={page * 20}&limit=20&status=P&sort=new_score"
            )
            html = client.get_html(url)
            page_rows = parse_comment_page(html, subject_id=subject_id, movie_title=movie.get("title", ""))
            comment_count += len(page_rows)
            pending_rows.extend(page_rows)

        merged = merge_csv_rows(output_csv, pending_rows, ["subject_id", "comment_user", "comment_text"], COMMENT_FIELDS)
        pending_rows = []
        save_checkpoint(
            checkpoint_name,
            {"next_movie_index": movie_index + 1, "comment_count": len(merged), "output_csv": str(output_csv)},
        )
        update_crawler_status(
            "crawl_comments",
            {
                "status": "running",
                "movie_index": movie_index + 1,
                "comment_count": len(merged),
                "last_movie": movie.get("title"),
                "last_movie_comments": comment_count,
                "output_csv": str(output_csv),
            },
        )

    merged_rows = []
    with output_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        merged_rows = list(csv.DictReader(csv_file))
    update_crawler_status(
        "crawl_comments",
        {"status": "completed", "comment_count": len(merged_rows), "movie_count": len(movies), "output_csv": str(output_csv)},
    )
    return {"output_csv": str(output_csv), "comment_count": len(merged_rows), "movie_count": len(movies)}


def build_behavior_dataset(
    movie_csv: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    user_count: int = 60,
    min_behaviors: int = 8,
    max_behaviors: int = 16,
    load_db: bool = True,
) -> Dict:
    movie_csv = movie_csv or (BASE_DIR / "datas.csv")
    output_csv = output_csv or (BASE_DIR / "behavior_dataset.csv")
    if not movie_csv.exists():
        raise FileNotFoundError(f"未找到电影数据文件: {movie_csv}")

    with movie_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        movies = [clean_movie_record(row) for row in csv.DictReader(csv_file)]
    if len(movies) < 10:
        raise ValueError("电影样本少于 10 条，建议先执行电影爬取。")

    genre_vocab = sorted({genre for movie in movies for genre in str(movie.get("genre", "")).split("/") if genre})
    users = []
    for idx in range(user_count):
        preferred = random.sample(genre_vocab, k=min(2, len(genre_vocab))) if genre_vocab else []
        users.append({"email": f"user_{idx + 1:03d}@example.com", "preferred_genres": preferred})

    rows = []
    for user in users:
        picked_movies = random.sample(movies, k=min(random.randint(min_behaviors, max_behaviors), len(movies)))
        for movie in picked_movies:
            movie_genres = set(str(movie.get("genre", "")).split("/"))
            preference_overlap = len(set(user["preferred_genres"]) & movie_genres)
            base_score = 6.5 + preference_overlap * 1.2 + random.uniform(-1.0, 1.2)
            score = max(1.0, min(10.0, round(base_score, 1)))
            behavior_type = 1 if random.random() < 0.7 else random.choice([2, 3])
            rows.append(
                {
                    "user_email": user["email"],
                    "movie_title": movie["title"],
                    "movie_subject_id": movie.get("subject_id", ""),
                    "behavior_type": behavior_type,
                    "score": score if behavior_type == 1 else "",
                    "create_time": f"2026-01-{random.randint(1, 28):02d} 12:{random.randint(0, 59):02d}:00",
                }
            )

    rows = dedupe_records(rows, ["user_email", "movie_subject_id", "behavior_type"])
    write_csv(output_csv, rows, BEHAVIOR_FIELDS)
    inserted_users = 0
    inserted_behaviors = 0
    if load_db:
        inserted_users, inserted_behaviors = _load_behaviors_into_db(rows)
    update_crawler_status(
        "build_behavior_dataset",
        {
            "status": "completed",
            "user_count": user_count,
            "behavior_count": len(rows),
            "output_csv": str(output_csv),
            "inserted_users": inserted_users,
            "inserted_behaviors": inserted_behaviors,
        },
    )
    return {
        "output_csv": str(output_csv),
        "user_count": user_count,
        "behavior_count": len(rows),
        "inserted_users": inserted_users,
        "inserted_behaviors": inserted_behaviors,
    }


def _load_behaviors_into_db(rows: List[Dict]) -> tuple[int, int]:
    init_db()
    db_movie_rows = querys("select id, title from movies", [], "select")
    db_movie_map = {title: movie_id for movie_id, title in db_movie_rows}
    known_users = {email for (email,) in querys("select email from user", [], "select")}
    inserted_users = 0
    inserted_behaviors = 0

    for row in rows:
        email = row["user_email"]
        if email not in known_users:
            querys(
                "insert into user(email, username, password) values(%s, %s, %s)",
                [email, email.split("@", 1)[0], generate_password_hash("123456")],
            )
            known_users.add(email)
            inserted_users += 1

        movie_id = db_movie_map.get(row["movie_title"])
        if not movie_id:
            continue
        inserted = behavior_service.save_behavior(
            user_email=email,
            movie_id=movie_id,
            behavior_type=int(row["behavior_type"]),
            score=float(row["score"]) if str(row["score"]).strip() else None,
            create_time=row["create_time"],
        )
        if inserted:
            inserted_behaviors += 1

    return inserted_users, inserted_behaviors


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawler jobs for the movie recommendation project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    movie_parser = subparsers.add_parser("movies", help="Crawl top movie pages.")
    movie_parser.add_argument("--pages", type=int, default=8)
    movie_parser.add_argument("--output", type=Path, default=BASE_DIR / "datas.csv")
    movie_parser.add_argument("--no-resume", action="store_true")

    comment_parser = subparsers.add_parser("comments", help="Crawl comment pages for movies in datas.csv.")
    comment_parser.add_argument("--movie-csv", type=Path, default=BASE_DIR / "datas.csv")
    comment_parser.add_argument("--output", type=Path, default=BASE_DIR / "comments_dataset.csv")
    comment_parser.add_argument("--pages-per-movie", type=int, default=3)
    comment_parser.add_argument("--limit-movies", type=int)
    comment_parser.add_argument("--no-resume", action="store_true")

    behavior_parser = subparsers.add_parser("behaviors", help="Generate a synthetic user behavior dataset.")
    behavior_parser.add_argument("--movie-csv", type=Path, default=BASE_DIR / "datas.csv")
    behavior_parser.add_argument("--output", type=Path, default=BASE_DIR / "behavior_dataset.csv")
    behavior_parser.add_argument("--user-count", type=int, default=60)
    behavior_parser.add_argument("--min-behaviors", type=int, default=8)
    behavior_parser.add_argument("--max-behaviors", type=int, default=16)
    behavior_parser.add_argument("--no-load-db", action="store_true")
    return parser


def main() -> int:
    parser = _build_cli()
    args = parser.parse_args()

    if args.command == "movies":
        result = crawl_top_movies(pages=args.pages, output_csv=args.output, resume=not args.no_resume)
        print(result)
        return 0
    if args.command == "comments":
        result = crawl_movie_comments(
            movie_csv=args.movie_csv,
            output_csv=args.output,
            pages_per_movie=args.pages_per_movie,
            limit_movies=args.limit_movies,
            resume=not args.no_resume,
        )
        print(result)
        return 0
    if args.command == "behaviors":
        result = build_behavior_dataset(
            movie_csv=args.movie_csv,
            output_csv=args.output,
            user_count=args.user_count,
            min_behaviors=args.min_behaviors,
            max_behaviors=args.max_behaviors,
            load_db=not args.no_load_db,
        )
        print(result)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
