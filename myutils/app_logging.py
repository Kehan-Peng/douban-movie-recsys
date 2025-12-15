from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(app) -> None:
    log_file = app.config["LOG_FILE"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if any(isinstance(handler, RotatingFileHandler) for handler in app.logger.handlers):
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = RotatingFileHandler(log_file, maxBytes=1_048_576, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
