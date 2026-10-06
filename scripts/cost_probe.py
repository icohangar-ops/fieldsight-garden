#!/usr/bin/env python
"""Tiny cost probe + full-run cost projection.

1. OFFLINE (always): renders every train/val/test example locally with the real cookbook
   renderers and counts tokens exactly (image tokens come from the Qwen HF image processor /
   tml-renderers; Tinker rejects requests whose expected image tokens don't match).
2. LIVE (unless --offline): ~5 val images -> Qwen3.6-35B-A3B sample (zero-shot prompt),
   the same 5 as training datums -> forward() on a fresh LoRA training client (billed at
   train rate, also verifies the image Datum path), and 2-3 val images -> Inkling sample.
   Uses VAL images only, never test.
3. Projects train + eval cost under the $10 budget and writes results/cost_probe.json.

Usage:
  python scripts/cost_probe.py                 # live probe (needs TINKER_API_KEY)
  python scripts/cost_probe.py --offline       # token accounting only, no API calls
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fieldsight.common import (  # noqa: E402
    INKLING, QWEN, build_generation_prompt, build_supervised_example, image_path,
    image_token_count, load_image, load_split, parse_label, parse_response_text, price_for,
    sample_cost, stratified_subsample, train_cost,
)

HARD_CAP_USD = 0.50  # refuse to run the live probe if its own worst case exceeds this


def offline_accounting():
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    acc = {}
    for split in ("train", "val", "test"):
        rows = load_split(split)
        tr, q_ft, q_zs, ink_zs, q_img, ink_img = [], [], [], [], [], []
        for r in rows:
            im = load_image(image_path(r))
            if split == "train":
                mi, w = build_supervised_example(QWEN, im, r["label"])
                d = datum_from_model_input_weights(mi, w, max_length=8192)
                tr.append(d.model_input.length)
            p_ft = build_generation_prompt(QWEN, im, "ft")
            p_zs = build_generation_prompt(QWEN, im, "zs")
            p_ink = build_generation_prompt(INKLING, im, "zs", inkling_effort=0.0)
            q_ft.append(p_ft.length); q_zs.append(p_zs.length); ink_zs.append(p_ink.length)
            q_img.append(image_token_count(p_ft)); ink_img.append(image_token_count(p_ink))
        s = {
            "n": len(rows),
            "qwen_image_tokens_mean": statistics.mean(q_img), "qwen_image_tokens_max": max(q_img),
            "inkling_image_tokens_mean": statistics.mean(ink_img),
            "qwen_ft_prompt_tokens_sum": sum(q_ft), "qwen_ft_prompt_tokens_mean": statistics.mean(q_ft),
            "qwen_zs_prompt_tokens_sum": sum(q_zs), "qwen_zs_prompt_tokens_mean": statistics.mean(q_zs),
            "inkling_zs_prompt_tokens_sum": sum(ink_zs), "inkling_zs_prompt_tokens_mean": statistics.mean(ink_zs),
        }
        if tr:
            s["train_datum_tokens_sum"] = sum(tr)
            s["train_datum_tokens_mean"] = statistics.mean(tr)
        acc[split] = s
        print(f"[offline] {split}: {json.dumps({k: round(v, 1) for k, v in s.items()})}")
    return acc


async def live_probe(n_qwen: int, n_inkling: int, inkling_effort: float, inkling_max_tokens: int):
    import tinker
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    rows = stratified_subsample(load_split("val"), max(n_qwen, n_inkling), seed=7)
    svc = tinker.ServiceClient()
    out = {"qwen_sample": [], "inkling_sample": [], "qwen_forward": {}}

    # worst-case guard
    worst = 0.0
    for r in rows[:n_qwen]:
        pl = build_generation_prompt(QWEN, load_image(image_path(r)), "zs").length
        worst += sample_cost(QWEN, pl, 0, 24) + pl * price_for(QWEN).train / 1e6
    for r in rows[:n_inkling]:
        pl = build_generation_prompt(INKLING, load_image(image_path(r)), "zs", inkling_effort).length
        worst += sample_cost(INKLING, pl, 0, inkling_max_tokens)
    print(f"[live] worst-case probe spend ${worst:.4f} (cap ${HARD_CAP_USD})")
    if worst > HARD_CAP_USD:
        raise SystemExit("probe worst case exceeds cap; lower --n-inkling / --inkling-max-tokens")
    out["worst_case_usd"] = worst

    async def run_sample(sc, model, r, prompt_mode, max_tokens, effort=0.0):
        im = load_image(image_path(r))
        prompt = build_generation_prompt(model, im, prompt_mode, inkling_effort=effort)
        from tinker_cookbook.renderers import get_renderer  # noqa: F401
        from fieldsight.common import get_renderer as gr
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                       stop=gr(model).get_stop_sequences())
        t0 = time.perf_counter()
        resp = await sc.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        dt = time.perf_counter() - t0
        toks = list(resp.sequences[0].tokens)
        text = parse_response_text(model, toks)
        strict, lenient = parse_label(text)
        rec = {"id": r["id"], "gold": r["label"], "text": text[:300], "pred_strict": strict,
               "pred_lenient": lenient, "prompt_tokens": prompt.length,
               "image_tokens": image_token_count(prompt), "cached_tokens": resp.prompt_cache_hit_tokens,
               "output_tokens": len(toks), "stop_reason": str(resp.sequences[0].stop_reason),
               "latency_s": round(dt, 3)}
        rec["usd"] = sample_cost(model, rec["prompt_tokens"], rec["cached_tokens"], rec["output_tokens"])
        print(f"[live] {model.split('/')[-1]:>14} {rec['latency_s']:6.2f}s p={rec['prompt_tokens']} "
              f"img={rec['image_tokens']} cache={rec['cached_tokens']} out={rec['output_tokens']} "
              f"gold={r['label']!r} -> {text[:80]!r}")
        return rec

    # Qwen sampling: sequential so latency is per-photo, not batched
    qsc = await svc.create_sampling_client_async(base_model=QWEN)
    for r in rows[:n_qwen]:
        out["qwen_sample"].append(await run_sample(qsc, QWEN, r, "zs", 24))

    # Qwen forward on training datums (verifies image SFT path + measures train-billed tokens)
    tc = await svc.create_lora_training_client_async(base_model=QWEN, rank=32)
    datums = []
    for r in rows[:n_qwen]:
        mi, w = build_supervised_example(QWEN, load_image(image_path(r)), r["label"])
        datums.append(datum_from_model_input_weights(mi, w, max_length=8192))
    t0 = time.perf_counter()
    fut = await tc.forward_async(datums, "cross_entropy")
    res = await fut.result_async()
    dt = time.perf_counter() - t0
    nll, ntok = 0.0, 0
    for d, o in zip(datums, res.loss_fn_outputs):
        lp = o["logprobs"].to_numpy() if hasattr(o["logprobs"], "to_numpy") else o["logprobs"]
        wt = d.loss_fn_inputs["weights"].to_numpy()
        nll += float(-(lp * wt).sum()); ntok += float(wt.sum())
    ftoks = sum(d.model_input.length for d in datums)
    out["qwen_forward"] = {"n": len(datums), "tokens": ftoks, "target_tokens": ntok,
                           "mean_target_nll": nll / max(ntok, 1), "latency_s": round(dt, 3),
                           "usd": train_cost(QWEN, ftoks), "metrics": res.metrics}
    print(f"[live] forward: {len(datums)} datums, {ftoks} tokens, base NLL/target-token "
          f"{nll / max(ntok, 1):.3f}, {dt:.2f}s")

    if n_inkling > 0:
        isc = await svc.create_sampling_client_async(base_model=INKLING)
        for r in rows[:n_inkling]:
            out["inkling_sample"].append(
                await run_sample(isc, INKLING, r, "zs", inkling_max_tokens, effort=inkling_effort))
    out["inkling_effort"] = inkling_effort
    out["probe_usd"] = (sum(x["usd"] for x in out["qwen_sample"] + out["inkling_sample"])
                        + out["qwen_forward"]["usd"])
    print(f"[live] estimated probe spend ${out['probe_usd']:.4f}")
    return out


def project(acc, live, epochs: int, inkling_n: int, n_val_evals: int, inkling_out_tokens: float,
            qwen_out_tokens: float, ft_out_tokens: float):
    p = {}
    tr = acc["train"]["train_datum_tokens_sum"]
    p["train_tokens_per_epoch"] = tr
    p["train_usd"] = train_cost(QWEN, tr * epochs)
    # in-training sampling evaluator on val (FT prompt, ~6 output tokens)
    v = acc["val"]
    p["val_evals_usd"] = n_val_evals * sample_cost(QWEN, v["qwen_ft_prompt_tokens_sum"], 0, v["n"] * ft_out_tokens)
    t = acc["test"]
    p["eval_base_qwen_usd"] = sample_cost(QWEN, t["qwen_zs_prompt_tokens_sum"], 0, t["n"] * qwen_out_tokens)
    p["eval_ft_qwen_usd"] = sample_cost(QWEN, t["qwen_ft_prompt_tokens_sum"], 0, t["n"] * ft_out_tokens)
    frac = inkling_n / t["n"]
    p["eval_inkling_usd"] = sample_cost(INKLING, t["inkling_zs_prompt_tokens_sum"] * frac, 0,
                                        inkling_n * inkling_out_tokens)
    p["eval_inkling_full_test_usd"] = sample_cost(INKLING, t["inkling_zs_prompt_tokens_sum"], 0,
                                                  t["n"] * inkling_out_tokens)
    p["checkpoint_storage_usd"] = 0.05  # LoRA rank-32 adapters, a few saves, $0.10/GB-month (rough)
    p["probe_usd"] = (live or {}).get("probe_usd", 0.0)
    p["total_usd"] = sum(v for k, v in p.items() if k.endswith("_usd") and k != "eval_inkling_full_test_usd")
    # per-1k-photo serving cost
    p["per_1k_photos_usd"] = {
        "base_qwen_zero_shot": 1000 * sample_cost(QWEN, t["qwen_zs_prompt_tokens_mean"], 0, qwen_out_tokens),
        "ft_qwen": 1000 * sample_cost(QWEN, t["qwen_ft_prompt_tokens_mean"], 0, ft_out_tokens),
        "inkling_zero_shot": 1000 * sample_cost(INKLING, t["inkling_zs_prompt_tokens_mean"], 0, inkling_out_tokens),
    }
    p["assumptions"] = {
        "epochs": epochs, "inkling_eval_n": inkling_n, "n_val_evals_during_training": n_val_evals,
        "qwen_zero_shot_output_tokens": qwen_out_tokens, "ft_output_tokens": ft_out_tokens,
        "inkling_output_tokens": inkling_out_tokens,
        "no_prefix_cache_discount_assumed": True,
        "prices_per_1M": {"qwen": vars(price_for(QWEN)), "inkling": vars(price_for(INKLING))},
    }
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-qwen", type=int, default=5)
    ap.add_argument("--n-inkling", type=int, default=3)
    ap.add_argument("--inkling-effort", type=float, default=0.0)
    ap.add_argument("--inkling-max-tokens", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--inkling-eval-n", type=int, default=100)
    ap.add_argument("--n-val-evals", type=int, default=6)
    args = ap.parse_args()

    acc = offline_accounting()
    live = None
    if not args.offline:
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("TINKER_API_KEY not set (use --offline for token accounting only)")
        live = asyncio.run(live_probe(args.n_qwen, args.n_inkling, args.inkling_effort,
                                      args.inkling_max_tokens))
    q_out = statistics.mean([x["output_tokens"] for x in live["qwen_sample"]]) if live else 8.0
    i_out = (statistics.mean([x["output_tokens"] for x in live["inkling_sample"]])
             if live and live["inkling_sample"] else 64.0)
    proj = project(acc, live, args.epochs, args.inkling_eval_n, args.n_val_evals, i_out, q_out, 8.0)
    print("[projection]", json.dumps(proj, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/cost_probe.json").write_text(
        json.dumps({"offline": acc, "live": live, "projection": proj}, indent=2, default=str))
    print("wrote results/cost_probe.json")


if __name__ == "__main__":
    main()
