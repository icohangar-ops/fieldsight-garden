#!/usr/bin/env python
"""Aggregate per-item eval outputs into results/report.json, results/RESULTS_tables.md and
results/chart.png. Usage:
  python scripts/report.py --items NAME=path/to/items_x.jsonl [...] --ft NAME
Every model is summarized on the full test set (if it has all 300 items) and on the fixed
100-image Inkling subset (stratified_subsample(test, 100, seed=0)); paired bootstrap CIs vs FT."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eval import paired_diff_ci, summarize  # noqa: E402
from fieldsight.common import load_split, stratified_subsample  # noqa: E402
from fieldsight.labels import CLASSES  # noqa: E402


def per_class_f1(items):
    from sklearn.metrics import f1_score
    g = [x["gold"] for x in items]; p = [x["pred_lenient"] or "<invalid>" for x in items]
    return dict(zip(CLASSES, f1_score(g, p, labels=CLASSES, average=None, zero_division=0).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--labels", nargs="*", default=[], help="NAME=Display label")
    ap.add_argument("--chart-labels", nargs="*", default=[], help="NAME=Short label (use \\n for newline)")
    a = ap.parse_args()
    items = {}
    for kv in a.items:
        k, v = kv.split("=", 1)
        items[k] = sorted([json.loads(l) for l in open(v)], key=lambda x: x["id"])
    disp = dict(kv.split("=", 1) for kv in a.labels)
    test = load_split("test")
    sub_ids = {r["id"] for r in stratified_subsample(test, 100, seed=0)}

    rows = {"full": [], "subset100": []}
    for k, it in items.items():
        if len(it) == len(test):
            s = summarize(k, "", it, 8); s["subset"] = "full"; rows["full"].append(s)
        sub = [x for x in it if x["id"] in sub_ids]
        assert len(sub) == 100, (k, len(sub))
        s = summarize(k, "", sub, 8); s["subset"] = "subset100"; rows["subset100"].append(s)
    for sset in rows.values():
        ft = next((s for s in sset if s["model"] == a.ft), None)
        for s in sset:
            s["cost_multiple_vs_ft"] = s["usd_per_1k_photos"] / ft["usd_per_1k_photos"] if ft else None

    paired = []
    for subset, ids in (("full", None), ("subset100", sub_ids)):
        ftc = {x["id"]: x["pred_lenient"] == x["gold"] for x in items[a.ft] if ids is None or x["id"] in ids}
        for k, it in items.items():
            if k == a.ft:
                continue
            oc = {x["id"]: x["pred_lenient"] == x["gold"] for x in it if ids is None or x["id"] in ids}
            if set(oc) != set(ftc):
                continue
            keys = sorted(ftc)
            d, lo, hi = paired_diff_ci(np.array([ftc[i] for i in keys], float), np.array([oc[i] for i in keys], float))
            paired.append({"subset": subset, "pair": f"{a.ft} - {k}", "acc_diff": d, "ci95": [lo, hi], "n": len(keys)})

    conf = {k: Counter((x["gold"], x["pred_lenient"] or "<invalid>") for x in it if x["pred_lenient"] != x["gold"]).most_common(8)
            for k, it in items.items()}
    f1s = {k: per_class_f1(it) for k, it in items.items() if len(it) == len(test)}
    report = {"rows": rows, "paired": paired, "top_confusions": conf, "per_class_f1": f1s}
    (ROOT / "results/report.json").write_text(json.dumps(report, indent=2))

    def table(sset):
        L = ["| model | n | accuracy [95% CI] | macro-F1 | invalid | p50 latency | p95 latency | $ / 1k photos | cost vs FT |",
             "|---|---|---|---|---|---|---|---|---|"]
        for s in sset:
            L.append(f"| {disp.get(s['model'], s['model'])} | {s['n']} | **{100*s['acc_lenient']:.1f}%** "
                     f"[{100*s['acc_lenient_ci95'][0]:.1f}, {100*s['acc_lenient_ci95'][1]:.1f}] | {s['macro_f1_lenient']:.3f} | "
                     f"{100*s['invalid_rate_lenient']:.1f}% | {s['latency_p50_s']:.2f} s | {s['latency_p95_s']:.2f} s | "
                     f"${s['usd_per_1k_photos']:.3f} | {s['cost_multiple_vs_ft']:.1f}× |")
        return "\n".join(L)
    md = ["## Full test set (300 images)", table(rows["full"]), "",
          "## Fixed 100-image subset (identical images for every model)", table(rows["subset100"]), "",
          "## Paired accuracy difference vs fine-tuned (bootstrap 95% CI)"]
    for p in paired:
        md.append(f"- {p['subset']}: {disp.get(p['pair'].split(' - ')[1], p['pair'])}: FT minus model = "
                  f"{100*p['acc_diff']:+.1f} pts [{100*p['ci95'][0]:+.1f}, {100*p['ci95'][1]:+.1f}] (n={p['n']})")
    (ROOT / "results/RESULTS_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sset = rows["subset100"]
    short = {k: v.replace("\\n", "\n") for k, v in (kv.split("=", 1) for kv in a.chart_labels)}
    names = [short.get(s["model"], disp.get(s["model"], s["model"])) for s in sset]
    acc = [100 * s["acc_lenient"] for s in sset]
    err = [[100 * (s["acc_lenient"] - s["acc_lenient_ci95"][0]) for s in sset],
           [100 * (s["acc_lenient_ci95"][1] - s["acc_lenient"]) for s in sset]]
    cost = [s["usd_per_1k_photos"] for s in sset]
    colors = ["#2e7d32" if s["model"] == a.ft else "#9e9e9e" for s in sset]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    b = ax[0].bar(names, acc, yerr=err, capsize=5, color=colors)
    ax[0].set_ylabel("Top-1 accuracy (%)"); ax[0].set_ylim(0, 105)
    ax[0].set_title("Accuracy, 100 identical test photos (95% CI)")
    ax[0].bar_label(b, labels=[f"{v:.0f}%" for v in acc], padding=14)
    b2 = ax[1].bar(names, cost, color=colors)
    ax[1].set_ylabel("USD per 1,000 photos"); ax[1].set_title("Cost per 1,000 photos (measured tokens)")
    ax[1].bar_label(b2, labels=[f"${v:.2f}" for v in cost], padding=3)
    ax[1].set_ylim(0, max(cost) * 1.2)
    for x in ax:
        x.tick_params(axis="x", labelsize=8.5)
        pass
    fig.suptitle("FieldSight Garden: PlantDoc leaf-disease ID (27 classes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "results/chart.png", dpi=150)
    print("wrote results/chart.png, results/report.json, results/RESULTS_tables.md")


if __name__ == "__main__":
    main()
