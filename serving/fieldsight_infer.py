"""Thin FieldSight Garden inference client for the Render app.

Depends only on `tinker` (which pulls `transformers` for the tokenizer) and `Pillow`; no torch,
no tinker-cookbook. Builds exactly the token sequence the cookbook renderer
`qwen3_5_disable_thinking` produced during training (verified token-for-token), sends it to the
fine-tuned LoRA via Tinker's native SamplingClient, and maps the answer to a 3-step fix.

Why not the OpenAI-compatible endpoint? It returned HTTP 400 "Image input is not supported for
this model" for this checkpoint on 2026-10-06, so image requests must use the native SDK.

    from serving.fieldsight_infer import FieldSight
    fs = FieldSight()                       # reads TINKER_API_KEY from the environment
    out = fs.diagnose(open("leaf.jpg", "rb").read())
    print(out["label"], out["steps"])
"""
from __future__ import annotations

import io
import json
import math
import os
import time
from pathlib import Path

import tinker
from PIL import Image, ImageOps

MODEL_PATH = os.environ.get(
    "FIELDSIGHT_MODEL_PATH",
    "tinker://f5090c0f-f54a-53c7-8aa7-e6486f065aef:train:0/sampler_weights/final",
)
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
MAX_SIDE = 512

# Token IDs for Qwen3.6 (identical to Qwen3.5 tokenizer), rendered by qwen3_5_disable_thinking:
#   "<|im_start|>user\n" + FT_PROMPT + "<|vision_start|>" [image] "<|vision_end|><|im_end|>"
#   "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PREFIX = [248045, 846, 198, 52811, 321, 8197, 303, 411, 15463, 6345, 30, 21134, 440, 279, 523,
          1132, 13, 248053]
SUFFIX = [248054, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
STOP = [248046]  # <|im_end|>

# Qwen2-VL-style image processor config for this model: patch 16, merge 2, pixels in [65536, 16777216]
FACTOR, MIN_PIXELS, MAX_PIXELS = 32, 65536, 16777216

_HERE = Path(__file__).resolve().parent
_FIXES = json.loads((_HERE.parent / "care" / "fixes.json").read_text())
CLASSES = sorted(_FIXES["fixes"])


def _smart_resize(h: int, w: int) -> tuple[int, int]:
    hb = max(FACTOR, round(h / FACTOR) * FACTOR)
    wb = max(FACTOR, round(w / FACTOR) * FACTOR)
    if hb * wb > MAX_PIXELS:
        beta = math.sqrt((h * w) / MAX_PIXELS)
        hb = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR)
        wb = max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    elif hb * wb < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        hb = math.ceil(h * beta / FACTOR) * FACTOR
        wb = math.ceil(w * beta / FACTOR) * FACTOR
    return hb, wb


def image_tokens(w: int, h: int) -> int:
    hb, wb = _smart_resize(h, w)
    return (hb // FACTOR) * (wb // FACTOR)


def preprocess(image_bytes: bytes) -> tuple[bytes, int]:
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")  # same default quality the cookbook renderer used in training
    return buf.getvalue(), image_tokens(im.width, im.height)


def build_prompt(jpeg: bytes, n_img_tokens: int) -> tinker.ModelInput:
    return tinker.ModelInput(chunks=[
        tinker.types.EncodedTextChunk(tokens=PREFIX),
        tinker.types.ImageChunk(data=jpeg, format="jpeg", expected_tokens=n_img_tokens),
        tinker.types.EncodedTextChunk(tokens=SUFFIX),
    ])


def _norm(s: str) -> str:
    return " ".join("".join(ch if ch.isalpha() else " " for ch in s.lower()).split())


class FieldSight:
    def __init__(self, model_path: str = MODEL_PATH):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.client = tinker.ServiceClient().create_sampling_client(model_path=model_path)
        self.params = tinker.SamplingParams(max_tokens=16, temperature=0.0, stop=STOP)

    def diagnose(self, image_bytes: bytes) -> dict:
        jpeg, n_img = preprocess(image_bytes)
        prompt = build_prompt(jpeg, n_img)
        t0 = time.perf_counter()
        resp = self.client.sample(prompt=prompt, num_samples=1, sampling_params=self.params).result()
        latency = time.perf_counter() - t0
        toks = [t for t in resp.sequences[0].tokens if t not in STOP]
        raw = self.tok.decode(toks, skip_special_tokens=True).strip()
        label = {_norm(c): c for c in CLASSES}.get(_norm(raw))
        fix = _FIXES["fixes"].get(label) if label else None
        return {
            "label": label, "raw": raw, "latency_s": round(latency, 3),
            "prompt_tokens": prompt.length, "output_tokens": len(resp.sequences[0].tokens),
            "type": fix["type"] if fix else None,
            "steps": fix["steps"] if fix else ["I couldn't tell from this photo. Try a closer, well-lit shot of one leaf."],
            "sources": [_FIXES["sources"][s] for s in fix["sources"]] if fix else [],
            "disclaimer": _FIXES["disclaimer"],
        }
