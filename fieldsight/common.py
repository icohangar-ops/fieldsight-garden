"""Shared helpers: data loading, prompt rendering for Qwen3.6 / Inkling, label parsing, pricing."""
from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from PIL import Image

from fieldsight.labels import CLASSES
from fieldsight.prompts import FT_PROMPT, ZS_PROMPT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS_JSON = Path("/workspace/hacktoberfest-tinker/models.json")

QWEN = "Qwen/Qwen3.6-35B-A3B"
QWEN_RENDERER = "qwen3_5_disable_thinking"
INKLING = "thinkingmachines/Inkling"
INKLING_RENDERER = "tml_v0"


# ---------------------------------------------------------------- data
def load_split(split: str) -> list[dict]:
    with open(DATA / "tinker" / f"{split}.jsonl") as fh:
        return [json.loads(line) for line in fh]


def load_image(rel_path: str) -> Image.Image:
    with Image.open(ROOT / rel_path) as im:
        return im.convert("RGB").copy()


def image_path(row: dict) -> str:
    return row["messages"][0]["content"][1]["image"]


def stratified_subsample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Deterministic, class-balanced-as-possible subsample (round-robin over shuffled classes)."""
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    order = sorted(by)
    out = []
    while len(out) < n:
        for c in order:
            if by[c] and len(out) < n:
                out.append(by[c].pop())
    return sorted(out, key=lambda r: r["id"])


# ---------------------------------------------------------------- rendering
@cache
def get_renderer(model_name: str):
    from tinker_cookbook.renderers import get_renderer as _gr
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    base = model_name.split(":", 1)[0]
    if base.startswith("thinkingmachines/Inkling"):
        return _gr(INKLING_RENDERER, get_tokenizer(base))
    from tinker_cookbook.image_processing_utils import get_image_processor

    return _gr(QWEN_RENDERER, get_tokenizer(base), image_processor=get_image_processor(base))


def user_message(image: Image.Image, prompt_mode: str):
    from tinker_cookbook.renderers import ImagePart, Message, TextPart

    text = FT_PROMPT if prompt_mode == "ft" else ZS_PROMPT
    # Text first, then image: keeps the (identical) text prefix cacheable across photos.
    return Message(role="user", content=[TextPart(type="text", text=text),
                                         ImagePart(type="image", image=image)])


def build_generation_prompt(model_name: str, image: Image.Image, prompt_mode: str,
                            inkling_effort: float = 0.0):
    r = get_renderer(model_name)
    msgs = [user_message(image, prompt_mode)]
    if model_name.startswith("thinkingmachines/Inkling"):
        return r.build_generation_prompt(msgs, effort=inkling_effort)
    return r.build_generation_prompt(msgs)


def build_supervised_example(model_name: str, image: Image.Image, label: str):
    from tinker_cookbook.renderers import Message, TextPart, TrainOnWhat

    r = get_renderer(model_name)
    msgs = [user_message(image, "ft"),
            Message(role="assistant", content=[TextPart(type="text", text=label)])]
    return r.build_supervised_example(msgs, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)


def image_token_count(model_input) -> int:
    return sum(c.length for c in model_input.chunks if "Image" in type(c).__name__)


def parse_response_text(model_name: str, tokens: list[int]) -> str:
    from tinker_cookbook.renderers import get_text_content

    msg = get_renderer(model_name).parse_response(tokens)[0]
    return get_text_content(msg)


# ---------------------------------------------------------------- label parsing
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_NORM_CLASSES = {_norm(c): c for c in CLASSES}


def parse_label(text: str) -> tuple[str | None, str | None]:
    """Return (strict, lenient) predictions.
    strict  = normalized output equals a class string exactly.
    lenient = strict, else a class string on the last non-empty line, else exactly one class
              string contained in the output, else closest class by difflib ratio >= 0.75.
              None if nothing (or more than one class) matches."""
    n = _norm(text)
    strict = _NORM_CLASSES.get(n)
    if strict:
        return strict, strict
    lines = [_norm(x) for x in text.strip().splitlines() if _norm(x)]
    for ln in reversed(lines):
        if ln in _NORM_CLASSES:
            return None, _NORM_CLASSES[ln]
    hits = [c for nc, c in _NORM_CLASSES.items() if re.search(rf"\b{re.escape(nc)}\b", n)]
    hits = [h for h in hits if not any(h != o and _norm(h) in _norm(o) for o in hits)]
    if len(hits) == 1:
        return None, hits[0]
    if len(hits) > 1:  # hedged answer naming several classes -> counted as no prediction
        return None, None
    cand = lines[-1] if lines else n
    m = difflib.get_close_matches(cand, list(_NORM_CLASSES), n=1, cutoff=0.75)
    return None, (_NORM_CLASSES[m[0]] if m else None)


# ---------------------------------------------------------------- pricing
@dataclass
class Price:
    prefill: float
    cached_prefill: float
    sample: float
    train: float


def _usd(s: str) -> float:
    return float(s.replace("$", ""))


@cache
def price_for(model_name: str) -> Price:
    """USD per 1M tokens from models.json. Fine-tuned LoRA checkpoints are priced as their
    base model (assumption; Tinker bills LoRA sampling at base-model rates)."""
    base = model_name.split(":", 1)[0] if not model_name.startswith("tinker://") else QWEN
    for m in json.loads(MODELS_JSON.read_text()):
        if m["tinker_id"] == base:
            return Price(_usd(m["prefill"]), _usd(m["cached_prefill"]), _usd(m["sample"]), _usd(m["train"]))
    raise KeyError(base)


def sample_cost(model_name: str, prompt_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    p = price_for(model_name)
    return ((prompt_tokens - cached_tokens) * p.prefill + cached_tokens * p.cached_prefill
            + output_tokens * p.sample) / 1e6


def train_cost(model_name: str, tokens: int) -> float:
    return tokens * price_for(model_name).train / 1e6
