#!/usr/bin/env python
"""Build the FieldSight Garden classification set from the official PlantDoc repo.

Steps: decode every image -> drop unusable classes -> find exact / near duplicates
(md5 + 64-bit dHash) -> drop label-conflicting duplicate groups, keep one image per
same-label group -> fixed stratified train/val/test split (seeded) -> resize to max
side 512 px JPEG -> write manifest, class list, stats, Tinker chat JSONL, and an
HF-datasets parquet copy (Image + ClassLabel) for the cookbook's ClassifierDataset.

Usage: python scripts/prepare_data.py [--raw raw/PlantDoc-Dataset] [--max-side 512]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fieldsight.labels import CLASSES, CLASS_TO_ID, DROPPED_FOLDERS, FOLDER_TO_LABEL  # noqa: E402
from fieldsight.prompts import FT_PROMPT  # noqa: E402

SEED = 20261006
TEST_TARGET = 300
VAL_TARGET = 150
DHASH_THRESHOLD = 4  # Hamming distance on 64-bit dHash considered a near-duplicate


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "raw/PlantDoc-Dataset"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--dhash-threshold", type=int, default=DHASH_THRESHOLD)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)

    records, bad, dropped = [], [], Counter()
    for orig_split in ("train", "test"):
        for cls_dir in sorted((raw / orig_split).iterdir()):
            if not cls_dir.is_dir():
                continue
            folder = cls_dir.name
            if folder not in FOLDER_TO_LABEL:
                raise SystemExit(f"Unknown PlantDoc folder: {folder}")
            for f in sorted(cls_dir.iterdir()):
                if folder in DROPPED_FOLDERS:
                    dropped[folder] += 1
                    continue
                data = f.read_bytes()
                try:
                    im = Image.open(io.BytesIO(data))
                    fmt = im.format
                    im = ImageOps.exif_transpose(im).convert("RGB")
                except Exception as e:  # noqa: BLE001
                    bad.append({"path": str(f.relative_to(raw)), "error": repr(e)})
                    continue
                label = FOLDER_TO_LABEL[folder][0]
                records.append({
                    "orig_path": str(f.relative_to(raw)), "orig_split": orig_split,
                    "folder": folder, "label": label, "orig_w": im.width, "orig_h": im.height,
                    "orig_format": fmt, "md5": hashlib.md5(data).hexdigest(), "dhash": dhash(im),
                    "_img": None,
                })
    print(f"decoded {len(records)} images, {len(bad)} unreadable, dropped {dict(dropped)}")

    # --- duplicate detection ---
    n = len(records)
    uf = UF(n)
    by_md5 = defaultdict(list)
    for i, r in enumerate(records):
        by_md5[r["md5"]].append(i)
    for idxs in by_md5.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    hashes = np.array([r["dhash"] for r in records], dtype=np.uint64)
    for i in range(n):
        x = np.bitwise_xor(hashes[i + 1:], hashes[i])
        # popcount
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for k in np.nonzero(d <= args.dhash_threshold)[0]:
            uf.union(i, i + 1 + int(k))
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    conflict_groups = [g for g in dup_groups if len({records[i]["label"] for i in g}) > 1]
    cross_split_groups = [g for g in dup_groups if len({records[i]["orig_split"] for i in g}) > 1]

    keep, removed = [], []
    for gid, g in enumerate(groups.values()):
        labels = {records[i]["label"] for i in g}
        for i in g:
            records[i]["dup_group"] = gid
            records[i]["dup_group_size"] = len(g)
        if len(labels) > 1:
            for i in g:
                removed.append({**records[i], "reason": "duplicate_with_conflicting_labels"})
            continue
        # keep the highest-resolution copy
        best = max(g, key=lambda i: records[i]["orig_w"] * records[i]["orig_h"])
        keep.append(best)
        for i in g:
            if i != best:
                removed.append({**records[i], "reason": "duplicate_same_label"})
    kept = [records[i] for i in keep]
    print(f"dup groups: {len(dup_groups)} (conflicting labels: {len(conflict_groups)}, "
          f"spanning orig train/test: {len(cross_split_groups)}); kept {len(kept)}, removed {len(removed)}")

    # --- stratified split ---
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for r in kept:
        by_label[r["label"]].append(r)
    N = len(kept)
    for label in CLASSES:
        rs = sorted(by_label[label], key=lambda r: r["orig_path"])
        rng.shuffle(rs)
        nc = len(rs)
        n_test = max(3, round(nc * TEST_TARGET / N))
        n_val = max(2, round(nc * VAL_TARGET / N))
        for k, r in enumerate(rs):
            r["split"] = "test" if k < n_test else ("val" if k < n_test + n_val else "train")

    # --- resize + write images ---
    img_root = out / "images"
    for r in kept:
        src = raw / r["orig_path"]
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        im.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)  # only shrinks
        slug = r["label"].replace(" ", "_")
        rid = f"{slug}__{r['md5'][:10]}"
        dst = img_root / r["split"] / slug / f"{rid}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="JPEG", quality=90, optimize=True)
        r.update(id=rid, path=str(dst.relative_to(ROOT)), w=im.width, h=im.height,
                 label_id=CLASS_TO_ID[r["label"]])

    kept.sort(key=lambda r: (r["split"], r["label"], r["id"]))
    fields = ["id", "split", "label", "label_id", "path", "w", "h", "orig_split", "orig_path",
              "folder", "orig_w", "orig_h", "orig_format", "md5", "dhash", "dup_group", "dup_group_size"]
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    with open(out / "removed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["reason"] + fields[7:] + ["label"], extrasaction="ignore")
        w.writeheader()
        w.writerows(removed)
    (out / "unreadable.json").write_text(json.dumps(bad, indent=2))

    # --- class list + stats ---
    counts = {c: Counter() for c in CLASSES}
    for r in kept:
        counts[r["label"]][r["split"]] += 1
    classes = []
    for c in CLASSES:
        folder = next(k for k, v in FOLDER_TO_LABEL.items() if v[0] == c)
        _, plant, cond, note = FOLDER_TO_LABEL[folder]
        classes.append({"id": CLASS_TO_ID[c], "label": c, "plant": plant, "condition": cond,
                        "plantdoc_folder": folder, "note": note,
                        "train": counts[c]["train"], "val": counts[c]["val"], "test": counts[c]["test"],
                        "total": sum(counts[c].values())})
    (out / "classes.json").write_text(json.dumps(classes, indent=2))
    tok_est = lambda r: max(1, round(r["w"] / 32)) * max(1, round(r["h"] / 32))  # noqa: E731
    split_sizes = Counter(r["split"] for r in kept)
    stats = {
        "source": "https://github.com/pratikkayal/PlantDoc-Dataset (commit 5467f60, 2021-05-02)",
        "license": "CC BY 4.0",
        "raw_images_decoded": len(records), "unreadable": len(bad), "dropped_classes": dict(dropped),
        "duplicate_groups": len(dup_groups), "duplicate_groups_conflicting_labels": len(conflict_groups),
        "duplicate_groups_spanning_original_train_test": len(cross_split_groups),
        "removed_images": Counter(r["reason"] for r in removed),
        "dhash_threshold": args.dhash_threshold,
        "kept_images": len(kept), "num_classes": len(CLASSES),
        "splits": dict(split_sizes), "seed": SEED, "max_side": args.max_side,
        "mean_resized_wh": [float(np.mean([r["w"] for r in kept])), float(np.mean([r["h"] for r in kept]))],
        "approx_qwen_image_tokens_mean": float(np.mean([tok_est(r) for r in kept])),
        "approx_qwen_image_tokens_p95": float(np.percentile([tok_est(r) for r in kept], 95)),
        "min_class_train": min(c["train"] for c in classes),
        "max_class_train": max(c["train"] for c in classes),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=int))

    # --- Tinker chat-format JSONL (cookbook Message/ImagePart schema; image = relative path) ---
    tdir = out / "tinker"
    tdir.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        with open(tdir / f"{split}.jsonl", "w") as fh:
            for r in kept:
                if r["split"] != split:
                    continue
                fh.write(json.dumps({
                    "id": r["id"], "label": r["label"], "label_id": r["label_id"],
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": FT_PROMPT},
                            {"type": "image", "image": r["path"]},
                        ]},
                        {"role": "assistant", "content": r["label"]},
                    ],
                }) + "\n")

    # --- HF datasets parquet (Image + ClassLabel), loadable with datasets.load_dataset(dir) ---
    import datasets
    feats = datasets.Features({"image": datasets.Image(), "label": datasets.ClassLabel(names=CLASSES),
                               "id": datasets.Value("string")})
    hdir = out / "hf"
    hdir.mkdir(exist_ok=True)
    for split, hf_split in (("train", "train"), ("val", "validation"), ("test", "test")):
        rows = [r for r in kept if r["split"] == split]
        ds = datasets.Dataset.from_dict(
            {"image": [{"bytes": (ROOT / r["path"]).read_bytes(), "path": None} for r in rows],
             "label": [r["label_id"] for r in rows], "id": [r["id"] for r in rows]}, features=feats)
        ds.to_parquet(str(hdir / f"{hf_split}-00000-of-00001.parquet"))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
