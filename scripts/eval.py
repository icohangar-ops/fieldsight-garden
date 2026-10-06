#!/usr/bin/env python
"""Evaluate base Qwen3.6-35B-A3B (zero-shot, class list in prompt), the fine-tuned LoRA
(short prompt) and Inkling (zero-shot, class list in prompt) on the fixed test split.

Metrics per model: accuracy (strict + lenient parsing) with 95% bootstrap CI, macro-F1,
invalid-output rate, mean/p50/p95 latency, $ per 1,000 photos and $ per correct answer
(from exact prompt/cached/output token counts x models.json prices). Also reports every
model on the Inkling subsample so the head-to-head comparison is on identical images,
with a paired bootstrap CI on the accuracy difference.

Usage:
  python scripts/eval.py --models base,ft,inkling --ft-path tinker://.../sampler_weights/final \
      --inkling-n 100
  python scripts/eval.py --models ft --ft-from-log runs/ft-qwen36-r32     # read final sampler path
  python scripts/eval.py --models base --limit 20                          # quick check
Outputs go to results/eval-<timestamp>/ (summary.json, summary.md, per-item JSONL, confusion CSVs).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fieldsight.common import (  # noqa: E402
    INKLING, QWEN, build_generation_prompt, get_renderer, image_path, image_token_count, load_image,
    load_split, parse_label, parse_response_text, sample_cost, stratified_subsample,
)
from fieldsight.labels import CLASSES  # noqa: E402

INVALID = "<invalid>"


def bootstrap_ci(correct: np.ndarray, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def macro_f1(gold, pred):
    from sklearn.metrics import f1_score
    return float(f1_score(gold, pred, labels=CLASSES, average="macro", zero_division=0))


def summarize(name, model_id, items, concurrency):
    gold = [x["gold"] for x in items]
    ok = [x for x in items if not x.get("error")]
    out = {"model": name, "model_id": model_id, "n": len(items), "errors": len(items) - len(ok),
           "concurrency": concurrency}
    for mode in ("strict", "lenient"):
        pred = [x[f"pred_{mode}"] or INVALID for x in items]
        c = np.array([p == g for p, g in zip(pred, gold)], dtype=float)
        lo, hi = bootstrap_ci(c)
        out[f"acc_{mode}"] = float(c.mean())
        out[f"acc_{mode}_ci95"] = [lo, hi]
        out[f"macro_f1_{mode}"] = macro_f1(gold, pred)
        out[f"invalid_rate_{mode}"] = float(np.mean([p == INVALID for p in pred]))
    lat = np.array([x["latency_s"] for x in ok]) if ok else np.array([np.nan])
    out.update(latency_mean_s=float(lat.mean()), latency_p50_s=float(np.percentile(lat, 50)),
               latency_p95_s=float(np.percentile(lat, 95)))
    for k in ("prompt_tokens", "image_tokens", "cached_tokens", "output_tokens"):
        out[f"mean_{k}"] = float(np.mean([x[k] for x in ok])) if ok else None
    usd = float(np.sum([x["usd"] for x in ok]))
    out["eval_usd"] = usd
    out["usd_per_1k_photos"] = 1000 * usd / max(len(ok), 1)
    out["usd_per_1k_correct"] = (1000 * usd / max(sum(x["pred_lenient"] == x["gold"] for x in ok), 1))
    return out


async def run_model(name, model_id, sampling_client, rows, prompt_mode, max_tokens, concurrency,
                    effort):
    import tinker
    base_for_render = INKLING if model_id.startswith("thinkingmachines/Inkling") else QWEN
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                   stop=get_renderer(base_for_render).get_stop_sequences())
    sem = asyncio.Semaphore(concurrency)
    price_id = model_id if not model_id.startswith("tinker://") else QWEN

    async def one(r):
        im = load_image(image_path(r))
        prompt = build_generation_prompt(base_for_render, im, prompt_mode, inkling_effort=effort)
        rec = {"id": r["id"], "gold": r["label"], "prompt_tokens": prompt.length,
               "image_tokens": image_token_count(prompt)}
        async with sem:
            t0 = time.perf_counter()
            for attempt in range(3):
                try:
                    resp = await sampling_client.sample_async(prompt=prompt, num_samples=1,
                                                              sampling_params=params)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        rec.update(error=repr(e)[:300], latency_s=time.perf_counter() - t0,
                                   cached_tokens=0, output_tokens=0, usd=0.0, text="",
                                   pred_strict=None, pred_lenient=None)
                        return rec
                    await asyncio.sleep(2 ** attempt)
            dt = time.perf_counter() - t0
        toks = list(resp.sequences[0].tokens)
        text = parse_response_text(base_for_render, toks)
        strict, lenient = parse_label(text)
        rec.update(text=text[:500], pred_strict=strict, pred_lenient=lenient,
                   cached_tokens=int(resp.prompt_cache_hit_tokens), output_tokens=len(toks),
                   stop_reason=str(resp.sequences[0].stop_reason), latency_s=dt)
        rec["usd"] = sample_cost(price_id, rec["prompt_tokens"], rec["cached_tokens"], rec["output_tokens"])
        return rec

    t0 = time.perf_counter()
    items = await asyncio.gather(*[one(r) for r in rows])
    print(f"[{name}] {len(items)} items in {time.perf_counter() - t0:.1f}s")
    return items


def confusion_csv(items, path):
    labels = CLASSES + [INVALID]
    m = {g: {p: 0 for p in labels} for g in CLASSES}
    for x in items:
        m[x["gold"]][x["pred_lenient"] or INVALID] += 1
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gold\\pred"] + labels)
        for g in CLASSES:
            w.writerow([g] + [m[g][p] for p in labels])


def ft_path_from_log(log_dir: str) -> str:
    p = Path(log_dir) if Path(log_dir).is_absolute() else ROOT / log_dir
    recs = [json.loads(line) for line in open(p / "checkpoints.jsonl")]
    recs = [r for r in recs if r.get("sampler_path")]
    return recs[-1]["sampler_path"]


async def amain(a):
    import tinker
    rows = load_split(a.split)
    if a.limit:
        rows = stratified_subsample(rows, a.limit, seed=a.seed)
    ink_rows = stratified_subsample(rows, a.inkling_n, seed=a.seed) if a.inkling_n else rows
    ink_ids = {r["id"] for r in ink_rows}
    svc = tinker.ServiceClient()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    outdir = ROOT / "results" / f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    outdir.mkdir(parents=True)
    all_items, summaries = {}, []
    for m in models:
        if m == "base":
            sc = await svc.create_sampling_client_async(base_model=QWEN)
            items = await run_model("base_qwen_zs", QWEN, sc, rows, "zs", a.max_tokens, a.concurrency, 0.0)
            mid = QWEN
        elif m == "ft":
            path = a.ft_path or ft_path_from_log(a.ft_from_log)
            sc = await svc.create_sampling_client_async(model_path=path)
            items = await run_model("ft_qwen", path, sc, rows, "ft", a.max_tokens, a.concurrency, 0.0)
            mid = path
        elif m == "inkling":
            sc = await svc.create_sampling_client_async(base_model=INKLING)
            items = await run_model(f"inkling_zs_effort{a.inkling_effort}", INKLING, sc, ink_rows, "zs",
                                    a.inkling_max_tokens, a.concurrency, a.inkling_effort)
            mid = INKLING
        else:
            raise SystemExit(f"unknown model {m}")
        name = {"base": "base_qwen_zs", "ft": "ft_qwen"}.get(m, f"inkling_zs_effort{a.inkling_effort}")
        all_items[name] = items
        with open(outdir / f"items_{name}.jsonl", "w") as fh:
            for x in items:
                fh.write(json.dumps(x) + "\n")
        confusion_csv(items, outdir / f"confusion_{name}.csv")
        s = summarize(name, mid, items, a.concurrency)
        s["subset"] = "inkling_subset" if m == "inkling" and a.inkling_n else "full"
        summaries.append(s)
        print(json.dumps(s, indent=2))

    # head-to-head on the Inkling subset (identical images)
    h2h = []
    if a.inkling_n and len(all_items) > 1:
        sub = {k: sorted([x for x in v if x["id"] in ink_ids], key=lambda x: x["id"])
               for k, v in all_items.items()}
        for k, v in sub.items():
            if not k.startswith("inkling"):
                s = summarize(k, "", v, a.concurrency); s["subset"] = "inkling_subset"; h2h.append(s)
        names = list(sub)
        corr = {k: np.array([x["pred_lenient"] == x["gold"] for x in v], float) for k, v in sub.items()}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if len(corr[names[i]]) == len(corr[names[j]]):
                    d, lo, hi = paired_diff_ci(corr[names[i]], corr[names[j]])
                    h2h.append({"pair": f"{names[i]} - {names[j]}", "acc_diff": d, "ci95": [lo, hi],
                                "n": int(len(corr[names[i]]))})
    result = {"split": a.split, "n_rows": len(rows), "inkling_n": len(ink_rows), "seed": a.seed,
              "inkling_effort": a.inkling_effort, "summaries": summaries, "head_to_head": h2h,
              "total_eval_usd": sum(s["eval_usd"] for s in summaries)}
    (outdir / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["| model | subset | n | acc (lenient) [95% CI] | acc strict | macro-F1 | invalid | p50 s | p95 s | mean s | $/1k photos | $/1k correct |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries + [h for h in h2h if "model" in h]:
        lines.append(f"| {s['model']} | {s['subset']} | {s['n']} | {s['acc_lenient']:.3f} "
                     f"[{s['acc_lenient_ci95'][0]:.3f}, {s['acc_lenient_ci95'][1]:.3f}] | {s['acc_strict']:.3f} | "
                     f"{s['macro_f1_lenient']:.3f} | {s['invalid_rate_lenient']:.3f} | {s['latency_p50_s']:.2f} | "
                     f"{s['latency_p95_s']:.2f} | {s['latency_mean_s']:.2f} | {s['usd_per_1k_photos']:.3f} | "
                     f"{s['usd_per_1k_correct']:.3f} |")
    for h in h2h:
        if "pair" in h:
            lines.append(f"\nPaired accuracy diff {h['pair']}: {h['acc_diff']:+.3f} "
                         f"[{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] (n={h['n']})")
    lines.append(f"\nTotal eval spend (estimated from token counts): ${result['total_eval_usd']:.4f}")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,ft,inkling")
    ap.add_argument("--ft-path", default=None, help="tinker://.../sampler_weights/...")
    ap.add_argument("--ft-from-log", default="runs/ft-qwen36-r32")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--limit", type=int, default=None, help="stratified subsample of the split")
    ap.add_argument("--inkling-n", type=int, default=100, help="0 = full split")
    ap.add_argument("--inkling-effort", type=float, default=0.0)
    ap.add_argument("--inkling-max-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set")
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
