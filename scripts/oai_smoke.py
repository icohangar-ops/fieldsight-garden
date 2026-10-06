#!/usr/bin/env python
"""Call the fine-tuned checkpoint through Tinker's OpenAI-compatible endpoint (beta) with one
val image, the way a thin Render app would (no torch / cookbook needed)."""
import base64, json, os, sys, time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fieldsight.prompts import FT_PROMPT  # noqa: E402

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
MODEL = sys.argv[1]
rows = [json.loads(l) for l in open(ROOT / "data/tinker/val.jsonl")][:: 25][:3]
client = OpenAI(base_url=BASE_URL, api_key=os.environ["TINKER_API_KEY"])
for r in rows:
    img = base64.b64encode((ROOT / r["messages"][0]["content"][1]["image"]).read_bytes()).decode()
    for extra in ({"chat_template_kwargs": {"enable_thinking": False}}, {}):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=16, temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": FT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}],
            extra_body=extra)
        print(json.dumps({"gold": r["label"], "extra_body": extra, "content": resp.choices[0].message.content,
                          "usage": resp.usage.model_dump() if resp.usage else None,
                          "latency_s": round(time.perf_counter() - t0, 2)}))
