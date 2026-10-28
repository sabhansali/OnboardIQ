from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_db_path() -> Path:
    raw_path = os.getenv("ONBOARDIQ_DB_PATH", "onboardiq.db")
    path = Path(raw_path)
    return path if path.is_absolute() else BASE_DIR / path


def load_config() -> dict[str, Any]:
    path = BASE_DIR / "config.json"
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config


def load_knowledge_base(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path is not None else BASE_DIR / "knowledge_base.json"
    with path.open("r", encoding="utf-8") as file:
        articles = json.load(file)
    if not isinstance(articles, list):
        raise ValueError("knowledge_base.json must contain a JSON array.")
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError("Each knowledge-base article must be a JSON object.")
        if not {"article_id", "title", "intent", "keywords", "approved", "response"}.issubset(article):
            raise ValueError(f"Knowledge article is missing required fields: {article!r}")
    return articles
