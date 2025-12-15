from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class AppConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "movie-recommendation-dev")
    JSON_AS_ASCII = False
    ADMIN_EMAILS = {
        item.strip().lower()
        for item in os.getenv("MOVIE_ADMIN_EMAILS", "alice@example.com").split(",")
        if item.strip()
    }
    LOG_DIR = BASE_DIR / "runtime" / "logs"
    LOG_FILE = LOG_DIR / "app.log"

