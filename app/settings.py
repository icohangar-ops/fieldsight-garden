"""Runtime settings. Secrets come from the environment and are never logged."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TRUE = {"1", "true", "yes", "on"}


def default_model_path() -> str:
    data = json.loads((ROOT / "results" / "serving.json").read_text())
    return str(data["model_path"])


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    mock: bool = False
    load_model: bool = True
    model_path: str = ""
    database_url: str | None = None
    max_bytes: int = 8_000_000
    rate_limit: int = 8
    rate_window_s: int = 60
    global_rate_limit: int = 60

    @classmethod
    def from_env(cls) -> Settings:
        mock = _flag("FIELDSIGHT_MOCK")
        url = os.environ.get("DATABASE_URL", "").strip() or None
        path = os.environ.get("FIELDSIGHT_MODEL_PATH", "").strip() or default_model_path()
        return cls(
            mock=mock,
            load_model=not mock,
            model_path=path,
            database_url=url,
            max_bytes=_int("FIELDSIGHT_MAX_BYTES", 8_000_000),
            rate_limit=_int("FIELDSIGHT_RATE_LIMIT", 8),
            rate_window_s=_int("FIELDSIGHT_RATE_WINDOW_S", 60),
            global_rate_limit=_int("FIELDSIGHT_GLOBAL_RATE_LIMIT", 60),
        )
