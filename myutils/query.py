import csv
import hashlib
import os
import sqlite3
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("MOVIE_DB_PATH", BASE_DIR / "doubanmovie.db"))
DATA_CSV_PATH = BASE_DIR / "datas.csv"
RL_MODEL_DIR = BASE_DIR / "artifacts" / "rl_models" / "ppo"

DEFAULT_USERS = [
    {"username": "alice", "email": "alice@example.com", "password": "123456"},
    {"username": "bob", "email": "bob@example.com", "password": "123456"},
    {"username": "carol", "email": "carol@example.com", "password": "123456"},
    {"username": "david", "email": "david@example.com", "password": "123456"},
    {"username": "emma", "email": "emma@example.com", "password": "123456"},
]

DEFAULT_BEHAVIOR_SEED = {
    "alice@example.com": [(0, 9.5), (3, 9.2), (6, 8.8)],
    "bob@example.com": [(0, 9.1), (4, 8.6), (7, 8.3)],
    "carol@example.com": [(2, 8.9), (3, 9.1), (5, 8.8)],
    "david@example.com": [(1, 9.4), (6, 8.5), (8, 8.7)],
    "emma@example.com": [(0, 8.7), (5, 9.0), (9, 9.2)],
}

MOVIE_METADATA = {
    "Inception": {"directors": "Christopher Nolan", "country": "美国/英国", "casts": "Leonardo DiCaprio/Joseph Gordon-Levitt/Elliot Page"},
    "The Shawshank Redemption": {"directors": "Frank Darabont", "country": "美国", "casts": "Tim Robbins/Morgan Freeman/Bob Gunton"},
    "The Godfather": {"directors": "Francis Ford Coppola", "country": "美国", "casts": "Marlon Brando/Al Pacino/James Caan"},
    "The Dark Knight": {"directors": "Christopher Nolan", "country": "美国/英国", "casts": "Christian Bale/Heath Ledger/Aaron Eckhart"},
    "Pulp Fiction": {"directors": "Quentin Tarantino", "country": "美国", "casts": "John Travolta/Samuel L. Jackson/Uma Thurman"},
    "Fight Club": {"directors": "David Fincher", "country": "美国", "casts": "Brad Pitt/Edward Norton/Helena Bonham Carter"},
    "Forrest Gump": {"directors": "Robert Zemeckis", "country": "美国", "casts": "Tom Hanks/Robin Wright/Gary Sinise"},
    "Interstellar": {"directors": "Christopher Nolan", "country": "美国/英国", "casts": "Matthew McConaughey/Anne Hathaway/Jessica Chastain"},
    "The Matrix": {"directors": "Lana Wachowski/Lilly Wachowski", "country": "美国/澳大利亚", "casts": "Keanu Reeves/Laurence Fishburne/Carrie-Anne Moss"},
    "The Lord of the Rings: The Return of the King": {"directors": "Peter Jackson", "country": "新西兰/美国", "casts": "Elijah Wood/Viggo Mortensen/Ian McKellen"},
}

SAMPLE_COMMENTS = {
    "Inception": ["剧情设定很烧脑，二刷依旧精彩。", "配乐和梦境层级设计都很惊艳。"],
    "The Shawshank Redemption": ["非常经典的励志电影。", "每次重看都有新的触动。"],
    "The Dark Knight": ["小丑塑造极具张力。"],
    "Interstellar": ["科幻和亲情结合得很出色。"],
}

_DB_INITIALIZED = False


def generate_password_hash(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"{b64encode(salt).decode()}${b64encode(digest).decode()}"


def check_password_hash(password_hash: str, password: str) -> bool:
    try:
        salt_b64, digest_b64 = password_hash.split("$", 1)
        salt = b64decode(salt_b64.encode())
        digest = b64decode(digest_b64.encode())
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return candidate == digest


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _convert_placeholders(sql: str) -> str:
    return sql.replace("%s", "?")


def querys(sql: str, params: Optional[Sequence[Any]] = None, query_type: str = "other"):
    params = list(params or [])
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(_convert_placeholders(sql), params)
        if query_type == "select":
            rows = cursor.fetchall()
            return [tuple(row) for row in rows]
        conn.commit()
        return cursor.lastrowid if cursor.lastrowid else cursor.rowcount
    finally:
        conn.close()


def init_db(force_seed: bool = False) -> None:
    global _DB_INITIALIZED
    if _DB_INITIALIZED and not force_seed:
        return

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS user (
                email TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                password TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                directors TEXT,
                types TEXT,
                country TEXT,
                casts TEXT,
                rate REAL DEFAULT 0,
                comment_len INTEGER DEFAULT 0,
                release_year INTEGER,
                duration INTEGER,
                summary TEXT,
                cover_url TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT,
                movie_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_email) REFERENCES user(email),
                FOREIGN KEY(movie_id) REFERENCES movies(id)
            );

            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                behavior_type INTEGER NOT NULL,
                score REAL,
                create_time TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_email) REFERENCES user(email),
                FOREIGN KEY(movie_id) REFERENCES movies(id),
                UNIQUE(user_email, movie_id, behavior_type)
            );

            CREATE TABLE IF NOT EXISTS rl_experience (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                behavior_type INTEGER NOT NULL,
                reward REAL NOT NULL,
                old_prob REAL,
                payload_json TEXT,
                status TEXT DEFAULT 'pending',
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_email) REFERENCES user(email),
                FOREIGN KEY(movie_id) REFERENCES movies(id)
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version_tag TEXT NOT NULL UNIQUE,
                storage_path TEXT NOT NULL,
                metrics_json TEXT,
                note TEXT,
                is_active INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                note TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_movies_types ON movies(types);
            CREATE INDEX IF NOT EXISTS idx_movies_directors ON movies(directors);
            CREATE INDEX IF NOT EXISTS idx_movies_country ON movies(country);
            CREATE INDEX IF NOT EXISTS idx_movies_casts ON movies(casts);
            CREATE INDEX IF NOT EXISTS idx_behavior_user_email ON user_behavior(user_email);
            CREATE INDEX IF NOT EXISTS idx_behavior_movie_id ON user_behavior(movie_id);
            CREATE INDEX IF NOT EXISTS idx_rl_experience_status ON rl_experience(status);
            CREATE INDEX IF NOT EXISTS idx_rl_experience_user_email ON rl_experience(user_email);
            CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(model_name, is_active);
            CREATE INDEX IF NOT EXISTS idx_experiment_runs_created_at ON experiment_runs(created_at);
            """
        )
        conn.commit()
        _seed_movies(conn, force_seed=force_seed)
        _seed_users(conn)
        _cleanup_invalid_relations(conn)
        _seed_behaviors(conn)
        _seed_comments(conn)
        _refresh_comment_count(conn)
        conn.commit()
        _DB_INITIALIZED = True
    finally:
        conn.close()


def _seed_movies(conn: sqlite3.Connection, force_seed: bool = False) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS total FROM movies")
    total = cursor.fetchone()[0]
    if total and not force_seed:
        return
    if force_seed:
        cursor.execute("DELETE FROM comments")
        cursor.execute("DELETE FROM rl_experience")
        cursor.execute("DELETE FROM model_registry")
        cursor.execute("DELETE FROM experiment_runs")
        cursor.execute("DELETE FROM user_behavior")
        cursor.execute("DELETE FROM movies")
        cursor.execute(
            "DELETE FROM sqlite_sequence WHERE name IN ('movies', 'comments', 'user_behavior', 'rl_experience', 'model_registry', 'experiment_runs')"
        )
        if RL_MODEL_DIR.exists():
            for model_file in RL_MODEL_DIR.glob("*.npz"):
                model_file.unlink()

    if not DATA_CSV_PATH.exists():
        return

    with DATA_CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            title = (row.get("title") or "").strip()
            metadata = MOVIE_METADATA.get(title, {})
            types = (row.get("genre") or "未知").replace(",", "/")
            cursor.execute(
                """
                INSERT INTO movies(title, directors, types, country, casts, rate, comment_len, release_year, duration, summary, cover_url)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    metadata.get("directors", "未知导演"),
                    types,
                    metadata.get("country", "未知"),
                    metadata.get("casts", "未知主演"),
                    float(row.get("rating") or 0),
                    0,
                    int(row.get("release_year") or 0),
                    int(row.get("duration") or 0),
                    (row.get("summary") or "").strip(),
                    None,
                ),
            )


def _seed_users(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user")
    if cursor.fetchone()[0]:
        return

    for user in DEFAULT_USERS:
        cursor.execute(
            "INSERT INTO user(email, username, password) VALUES(?, ?, ?)",
            (
                user["email"],
                user["username"],
                generate_password_hash(user["password"]),
            ),
        )


def _seed_comments(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM comments")
    if cursor.fetchone()[0]:
        return

    cursor.execute("SELECT id, title FROM movies")
    movie_map = {row[1]: row[0] for row in cursor.fetchall()}
    cursor.execute("SELECT email FROM user ORDER BY email")
    users = [row[0] for row in cursor.fetchall()] or ["system@example.com"]

    user_index = 0
    for title, comments in SAMPLE_COMMENTS.items():
        movie_id = movie_map.get(title)
        if not movie_id:
            continue
        for comment in comments:
            cursor.execute(
                "INSERT INTO comments(user_email, movie_id, content) VALUES(?, ?, ?)",
                (users[user_index % len(users)], movie_id, comment),
            )
            user_index += 1


def _cleanup_invalid_relations(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM user_behavior
        WHERE movie_id NOT IN (SELECT id FROM movies)
           OR user_email NOT IN (SELECT email FROM user)
        """
    )
    cursor.execute(
        """
        DELETE FROM rl_experience
        WHERE movie_id NOT IN (SELECT id FROM movies)
           OR user_email NOT IN (SELECT email FROM user)
        """
    )


def _seed_behaviors(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_behavior")
    if cursor.fetchone()[0]:
        return

    cursor.execute("SELECT id FROM movies ORDER BY id")
    movie_ids = [row[0] for row in cursor.fetchall()]
    if not movie_ids:
        return

    create_time = "2026-01-01 12:00:00"
    for user_email, behaviors in DEFAULT_BEHAVIOR_SEED.items():
        for movie_index, score in behaviors:
            if movie_index >= len(movie_ids):
                continue
            cursor.execute(
                """
                INSERT INTO user_behavior(user_email, movie_id, behavior_type, score, create_time)
                VALUES(?, ?, 1, ?, ?)
                """,
                (user_email, movie_ids[movie_index], score, create_time),
            )


def _refresh_comment_count(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("UPDATE movies SET comment_len = 0")
    cursor.execute(
        """
        UPDATE movies
        SET comment_len = (
            SELECT COUNT(1)
            FROM comments
            WHERE comments.movie_id = movies.id
        )
        """
    )


def _row_to_movie(row: sqlite3.Row) -> Dict[str, Any]:
    movie = dict(row)
    movie["rating"] = movie.get("rate", 0)
    movie["poster_url"] = movie.get("cover_url")
    movie["release_date"] = movie.get("release_year")
    movie["genres"] = [item for item in (movie.get("types") or "").split("/") if item]
    movie["types_list"] = movie["genres"]
    movie["directors_list"] = [item for item in (movie.get("directors") or "").split("/") if item]
    movie["country_list"] = [item for item in (movie.get("country") or "").split("/") if item]
    movie["casts_list"] = [item for item in (movie.get("casts") or "").split("/") if item]
    movie["description"] = movie.get("summary") or ""
    return movie


def get_movie_data(movie_id: Optional[int] = None) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if movie_id is None:
            cursor.execute("SELECT * FROM movies ORDER BY rate DESC, comment_len DESC")
            return [_row_to_movie(row) for row in cursor.fetchall()]
        cursor.execute("SELECT * FROM movies WHERE id = ?", (movie_id,))
        row = cursor.fetchone()
        return _row_to_movie(row) if row else None
    finally:
        conn.close()


def get_top_movies(num_movies: int = 25) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM movies ORDER BY rate DESC, comment_len DESC, title ASC LIMIT ?",
            (num_movies,),
        )
        return [_row_to_movie(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def search_movies(query: str) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        keyword = f"%{query.strip()}%"
        cursor.execute(
            """
            SELECT * FROM movies
            WHERE title LIKE ? OR types LIKE ? OR directors LIKE ? OR country LIKE ?
            ORDER BY rate DESC, comment_len DESC
            """,
            (keyword, keyword, keyword, keyword),
        )
        return [_row_to_movie(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_movie_comments(movie_id: int) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.id, c.content, c.created_at, COALESCE(u.username, c.user_email, '匿名用户') AS username
            FROM comments c
            LEFT JOIN user u ON u.email = c.user_email
            WHERE c.movie_id = ?
            ORDER BY c.created_at DESC, c.id DESC
            """,
            (movie_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def create_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM user WHERE email = ?", (email,))
        if cursor.fetchone():
            return False, "该邮箱已注册。"
        cursor.execute(
            "INSERT INTO user(email, username, password) VALUES(?, ?, ?)",
            (email, username, generate_password_hash(password)),
        )
        conn.commit()
        return True, "注册成功，请登录。"
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT email, username, password FROM user WHERE email = ?", (email,))
        row = cursor.fetchone()
        if row and check_password_hash(row["password"], password):
            return {"email": row["email"], "username": row["username"]}
        return None
    finally:
        conn.close()


def get_user_preferences(user_email: str) -> Dict[int, float]:
    return get_user_ratings(user_email)


def get_user_ratings(user_email: str) -> Dict[int, float]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT movie_id, score FROM user_behavior WHERE user_email = ? AND behavior_type = 1",
            (user_email,),
        )
        return {row[0]: row[1] for row in cursor.fetchall() if row[1] is not None}
    finally:
        conn.close()


def get_movie_ratings(movie_id: int) -> List[float]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT score FROM user_behavior WHERE movie_id = ? AND behavior_type = 1 AND score IS NOT NULL",
            (movie_id,),
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_all_movie_ids() -> List[int]:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM movies ORDER BY id")
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
