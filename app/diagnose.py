"""Turn one leaf photo into the public diagnosis payload.

Mock mode reads care/fixes.json and does not import tinker or transformers.
Real mode calls serving/fieldsight_infer.py, which owns the training prompt.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.settings import ROOT
from fieldsight.labels import FOLDER_TO_LABEL

logger = logging.getLogger("fieldsight")

Image.MAX_IMAGE_PIXELS = 24_000_000

_FIXES = json.loads((ROOT / "care" / "fixes.json").read_text())
CLASSES: list[str] = sorted(_FIXES["fixes"])
DISCLAIMER: str = _FIXES["disclaimer"]

LABEL_META: dict[str, tuple[str, str]] = {
    plant_label: (plant, condition)
    for plant_label, plant, condition, _note in FOLDER_TO_LABEL.values()
    if plant_label
}


class BadImage(Exception):
    pass


class TooLarge(Exception):
    pass


class ModelNotReady(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def title_phrase(text: str) -> str:
    return " ".join(part.capitalize() for part in text.split())


def presentation(label: str | None) -> tuple[str | None, str | None]:
    if not label:
        return None, None
    meta = LABEL_META.get(label)
    if meta is None:
        return title_phrase(label), None
    plant, condition = meta
    return title_phrase(plant), title_phrase(condition)


def public_sources(sources: list[dict] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for source in sources or []:
        url = str(source.get("url") or "")
        if not (url.startswith("https://") or url.startswith("http://")):
            continue
        out.append({
            "title": str(source.get("title") or ""),
            "publisher": str(source.get("publisher") or ""),
            "url": url,
        })
    return out


def shape_result(raw: dict[str, Any], *, mock: bool) -> dict[str, Any]:
    label = raw.get("label")
    plant, condition = presentation(label if isinstance(label, str) else None)
    confidence = raw.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = None
    steps = raw.get("steps") or []
    return {
        "ok": bool(label),
        "label": label if isinstance(label, str) else None,
        "plant": plant,
        "condition": condition,
        "type": raw.get("type"),
        "confidence": confidence,
        "steps": [str(step) for step in steps],
        "sources": public_sources(raw.get("sources")),
        "disclaimer": raw.get("disclaimer") or DISCLAIMER,
        "latency_s": raw.get("latency_s"),
        "mock": mock,
    }


def result_for_label(label: str, *, latency_s: float, mock: bool, raw_text: str | None = None) -> dict[str, Any]:
    fix = _FIXES["fixes"].get(label)
    if fix is None:
        return shape_result({
            "label": None,
            "raw": raw_text,
            "latency_s": latency_s,
            "type": None,
            "steps": ["I couldn't tell from this photo. Try a closer, well-lit shot of one leaf."],
            "sources": [],
            "disclaimer": DISCLAIMER,
        }, mock=mock)
    return shape_result({
        "label": label,
        "latency_s": latency_s,
        "type": fix["type"],
        "steps": fix["steps"],
        "sources": [_FIXES["sources"][key] for key in fix["sources"]],
        "disclaimer": DISCLAIMER,
        "confidence": None,
    }, mock=mock)


def validate_image(data: bytes, *, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise TooLarge()
    if not data:
        raise BadImage("empty")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            width, height = im.size
    except Image.DecompressionBombError as exc:
        raise TooLarge() from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise BadImage("unreadable") from exc
    if width < 1 or height < 1:
        raise BadImage("empty")


async def read_limited(upload: Any, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        block = await upload.read(64 * 1024)
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            raise TooLarge()
        chunks.append(block)
    return b"".join(chunks)


class MockEngine:
    ready = True
    failed = False
    detail: str | None = None

    def diagnose(self, image_bytes: bytes, *, max_bytes: int) -> dict[str, Any]:
        validate_image(image_bytes, max_bytes=max_bytes)
        digest = hashlib.sha256(image_bytes).digest()
        label = CLASSES[int.from_bytes(digest[:8], "big") % len(CLASSES)]
        return result_for_label(label, latency_s=0.01, mock=True, raw_text=label)


class RealEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ready = False
        self.failed = False
        self.detail: str | None = None
        self._fs: Any = None
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._load, name="fieldsight-model", daemon=True).start()

    def _load(self) -> None:
        try:
            if not os.environ.get("TINKER_API_KEY", "").strip():
                raise RuntimeError("TINKER_API_KEY is not set")
            from serving.fieldsight_infer import FieldSight

            client = FieldSight(model_path=self.model_path)
            with self._lock:
                self._fs = client
                self.ready = True
                self.failed = False
                self.detail = None
            logger.info("sampling client ready")
        except Exception:
            logger.exception("model load failed")
            missing_key = not os.environ.get("TINKER_API_KEY", "").strip()
            with self._lock:
                self.ready = False
                self.failed = True
                self.detail = (
                    "Set TINKER_API_KEY on the service, then redeploy."
                    if missing_key
                    else "The garden model failed to load. Check the deploy logs."
                )

    def diagnose(self, image_bytes: bytes, *, max_bytes: int) -> dict[str, Any]:
        validate_image(image_bytes, max_bytes=max_bytes)
        with self._lock:
            ready = self.ready
            failed = self.failed
            detail = self.detail
            client = self._fs
        if failed:
            raise ModelNotReady(detail or "The garden model is unavailable.")
        if not ready or client is None:
            raise ModelNotReady("The model is still waking up. Try this photo again in a moment.")
        raw = client.diagnose(image_bytes)
        return shape_result(raw, mock=False)
