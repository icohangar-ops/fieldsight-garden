# FieldSight Garden: data and harness

DEV Hacktoberfest 2026, Week 1 ("Touch Grass"). One leaf photo goes in, and the app returns the plant, the disease and a 3-step fix read aloud.
This folder holds the data prep, the Tinker scripts (cost probe, LoRA train, eval) and the fix lookup table.
Model under test: `Qwen/Qwen3.6-35B-A3B` with a LoRA adapter, renderer `qwen3_5_disable_thinking`. Baselines: the same model untuned (zero-shot) and `thinkingmachines/Inkling` (zero-shot).

## Dataset: PlantDoc (cropped classification release)
- **Source:** https://github.com/pratikkayal/PlantDoc-Dataset, commit `5467f60` (2021-05-02), shallow-cloned to `raw/PlantDoc-Dataset`. Note: `github.com/pantheon/PlantDoc-Dataset` returns 404. pratikkayal is the official repo linked from the paper.
- **License:** Creative Commons Attribution 4.0 International (`raw/PlantDoc-Dataset/LICENSE.txt`). Redistribution and derivatives are allowed with attribution. The images were scraped from the web by the authors, and some carry stock-photo watermarks (e.g. Alamy). CC BY 4.0 is the dataset's license; we have not verified rights for each individual image.
- **Citation:**
```bibtex
@inproceedings{10.1145/3371158.3371196,
  author = {Singh, Davinder and Jain, Naman and Jain, Pranjali and Kayal, Pratik and Kumawat, Sudhakar and Batra, Nipun},
  title = {PlantDoc: A Dataset for Visual Plant Disease Detection},
  year = {2020}, publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 7th ACM IKDD CoDS and 25th COMAD}, pages = {249--253},
  doi = {10.1145/3371158.3371196}, url = {https://doi.org/10.1145/3371158.3371196}
}
```
  arXiv: https://arxiv.org/abs/1911.10317

### Cleaning (`scripts/prepare_data.py`, deterministic, seed 20261006)
1. All 2,576 files decoded. None were unreadable (4 were really PNG and 2 MPO despite the .jpg name). EXIF orientation was applied and images converted to RGB.
2. **Dropped** `Tomato two spotted spider mites leaf`: only 2 images, none in test.
3. **Duplicates:** exact md5 match plus 64-bit dHash with Hamming distance ≤ 4, grouped by union-find. This found 66 groups. Spot-checks confirmed they are the same photo at different sizes or watermarks.
   - **32 groups carry conflicting labels**, i.e. the same photo filed under two classes, mostly early/late blight across potato and tomato, corn gray leaf spot vs leaf blight, and septoria vs bacterial spot. All 66 images in those groups were removed.
   - For the 34 same-label groups, the highest-resolution copy was kept.
   - 14 groups straddled PlantDoc's original train/test split, which means the original test set leaks.
   - Details: `data/removed.csv`.
4. **Re-split** into a fixed stratified train/val/test split (2,026 / 150 / 300). This **is not the paper's split**: the original test set had leakage and only 236 images. `orig_split` is kept in `data/manifest.csv` (24 of our 300 test images came from the original test set).
5. Resized so the longest side is at most **512 px** (LANCZOS, JPEG q90, never upscaled). 177 originals have a short side under 256 px. Qwen's image processor upsamples those to its 65,536-pixel minimum.

**Result:** 2,476 images, 27 classes, 13 species.
- Train per class: 44–149. Test per class: 6–22.
- Mean resized size: 456×387.
- Image tokens: Qwen3.6 averages 173 (max 256); Inkling averages 118.

| id | label (model target) | PlantDoc folder | train | val | test | total |
|---|---|---|---|---|---|---|
| 0 | apple healthy | Apple leaf | 74 | 6 | 11 | 91 |
| 1 | apple rust | Apple rust leaf | 72 | 5 | 11 | 88 |
| 2 | apple scab | Apple Scab Leaf | 75 | 6 | 11 | 92 |
| 3 | bell pepper healthy | Bell_pepper leaf | 50 | 4 | 7 | 61 |
| 4 | bell pepper leaf spot | Bell_pepper leaf spot | 55 | 4 | 8 | 67 |
| 5 | blueberry healthy | Blueberry leaf | 95 | 7 | 14 | 116 |
| 6 | cherry healthy | Cherry leaf | 47 | 3 | 7 | 57 |
| 7 | corn gray leaf spot | Corn Gray leaf spot | 52 | 4 | 8 | 64 |
| 8 | corn leaf blight | Corn leaf blight | 149 | 11 | 22 | 182 |
| 9 | corn rust | Corn rust leaf | 94 | 7 | 14 | 115 |
| 10 | grape black rot | grape leaf black rot | 51 | 4 | 8 | 63 |
| 11 | grape healthy | grape leaf | 57 | 4 | 8 | 69 |
| 12 | peach healthy | Peach leaf | 91 | 7 | 13 | 111 |
| 13 | potato early blight | Potato leaf early blight | 83 | 6 | 12 | 101 |
| 14 | potato late blight | Potato leaf late blight | 77 | 6 | 11 | 94 |
| 15 | raspberry healthy | Raspberry leaf | 97 | 7 | 14 | 118 |
| 16 | soybean healthy | Soyabean leaf | 53 | 4 | 8 | 65 |
| 17 | squash powdery mildew | Squash Powdery mildew leaf | 104 | 8 | 16 | 128 |
| 18 | strawberry healthy | Strawberry leaf | 78 | 6 | 12 | 96 |
| 19 | tomato bacterial spot | Tomato leaf bacterial spot | 86 | 6 | 13 | 105 |
| 20 | tomato early blight | Tomato Early blight leaf | 61 | 5 | 9 | 75 |
| 21 | tomato healthy | Tomato leaf | 51 | 4 | 8 | 63 |
| 22 | tomato late blight | Tomato leaf late blight | 82 | 6 | 12 | 100 |
| 23 | tomato leaf mold | Tomato mold leaf | 74 | 5 | 11 | 90 |
| 24 | tomato mosaic virus | Tomato leaf mosaic virus | 44 | 3 | 6 | 53 |
| 25 | tomato septoria leaf spot | Tomato Septoria leaf spot | 115 | 8 | 17 | 140 |
| 26 | tomato yellow leaf curl virus | Tomato leaf yellow virus | 59 | 4 | 9 | 72 |
| | **total** | | **2026** | **150** | **300** | **2476** |

### Label-noise caveats (put these in the write-up)
- PlantDoc was annotated from web-scraped images. Beyond the 32 exact-duplicate conflicts we removed, near-identical but not duplicate images with different labels almost certainly remain. The paper's own baselines reach only around 70% accuracy.
- Several pairs are visually confusable and may be mislabeled even by experts: potato vs tomato early blight, early vs late blight, septoria vs bacterial spot vs early blight, and corn gray leaf spot vs leaf blight.
- Label names are our cleanup of PlantDoc's folder names (see `fieldsight/labels.py`). Assumed mappings:
  - "Tomato leaf yellow virus" → `tomato yellow leaf curl virus`
  - "Corn leaf blight" ≈ northern corn leaf blight
  - "Corn rust" ≈ common rust
  - "Bell_pepper leaf spot" ≈ bacterial spot
  - "Apple rust" ≈ cedar-apple rust
- "Healthy" classes are really "no labeled disease". Many photos show whole plants, fruit or backgrounds rather than a single leaf.
- The test set is small (300 images, 6–22 per class). Always report the 95% bootstrap CI. Per-class F1 is noisy.

## Files
```
fieldsight-garden/
  README.md                  this file (data card, commands, cost math)
  requirements.lock.txt      uv pip freeze of .venv (Python 3.13.5)
  raw/PlantDoc-Dataset/      official clone (CC BY 4.0), untouched
  data/
    images/{train,val,test}/<class_slug>/<class_slug>__<md5[:10]>.jpg   resized, ≤512px
    manifest.csv             one row per kept image: split, label, label_id, path, size, orig path/split, md5, dHash, dup group
    removed.csv              100 removed duplicates with reason
    classes.json             class list with plant/condition, PlantDoc folder, notes, per-split counts
    stats.json               dataset stats
    tinker/{train,val,test}.jsonl   chat-format SFT rows (see below)
    hf/{train,validation,test}-*.parquet   HF datasets copy (Image + ClassLabel), works with datasets.load_dataset("data/hf")
  fieldsight/
    labels.py                PlantDoc folder → canonical label map, CLASSES
    prompts.py               FT_PROMPT (short) and ZS_PROMPT (short + class list)
    common.py                renderers (Qwen / Inkling), prompt building, label parsing, pricing from models.json
  scripts/
    prepare_data.py          builds everything under data/
    cost_probe.py            offline exact token accounting + tiny live probe + cost projection
    train.py                 LoRA SFT via tinker_cookbook.supervised.train
    eval.py                  base vs FT vs Inkling: acc + CI, macro-F1, latency, $/1k photos
  care/
    build_fixes.py, fixes.json   3-step fix per class with extension-service sources
  results/
    cost_probe.json, cost_probe.log
```

### Training format
Each line of `data/tinker/*.jsonl` uses the cookbook `Message`/`ImagePart` schema, which is what `tinker_cookbook.recipes.vlm_classifier` builds internally. The image path is relative to the project root.
```json
{"id": "...", "label": "tomato early blight", "label_id": 20,
 "messages": [{"role": "user", "content": [{"type": "text", "text": "Plant and disease in this leaf photo? Answer with the class only."},
                                         {"type": "image", "image": "data/images/train/.../x.jpg"}]},
              {"role": "assistant", "content": "tomato early blight"}]}
```
- `scripts/train.py` loads each image as a PIL image and calls `renderer.build_supervised_example(..., TrainOnWhat.LAST_ASSISTANT_MESSAGE)`, then `datum_from_model_input_weights`.
- Only the class string plus `<|im_end|>` is trained. We checked this by decoding the weighted target tokens.
- Text comes before the image, so the shared text prefix is cacheable. The live probe saw 128 cached prompt tokens per call.
- Zero-shot baselines get `ZS_PROMPT`, which adds the closed list of 27 classes. The fine-tuned model is trained and served with the short prompt only. That makes its prompt about 150 tokens shorter, part of the cost win.

## Commands
```bash
cd /workspace/fieldsight-garden && source .venv/bin/activate      # TINKER_API_KEY must be set

# 0. (re)build data (~2 min)
python scripts/prepare_data.py

# 1. cost probe: 5 val images on Qwen3.6 (sample) + 5-datum forward on a LoRA client + 3 on Inkling
python scripts/cost_probe.py                     # live (hard cap $0.50 inside; actual ≈ $0.004)
python scripts/cost_probe.py --offline           # token accounting + projection only

# 2. train (projected ≈ $1.47; aborts if projection > max_usd=3)
python scripts/train.py dry_run=True                                     # print projected cost only
python scripts/train.py log_dir=runs/smoke max_steps=2 eval_every=0 save_every=0   # optional ~$0.02 smoke
python scripts/train.py log_dir=runs/ft-qwen36-r32                       # full: r32, LR 5e-4 cosine, bs32, 3 ep (192 steps)

# 3. eval (writes results/eval-<ts>/summary.md|json, per-item jsonl, confusion CSVs)
python scripts/eval.py --models base --limit 30                          # cheap sanity check (~$0.006)
python scripts/eval.py --models base,ft,inkling --ft-from-log runs/ft-qwen36-r32 --inkling-n 100
#   or --ft-path tinker://<run>/sampler_weights/final ; --inkling-n 0 = full 300 (~$0.17)
#   --inkling-effort 0.0 (default; ~6 output tokens). Try 0.5 for a "thinking" Inkling column.
```

## Cost math (prices from /workspace/hacktoberfest-tinker/models.json, USD per 1M tokens)
- **Qwen3.6-35B-A3B:** prefill $0.54, cached prefill $0.108, sample $1.335, train $1.177.
- **Inkling** (50% promo): prefill $1.87, cached prefill $0.374, sample $4.68, train $5.61.

Token counts are **exact**. They are rendered locally with the same renderers and image processors Tinker uses, and Tinker rejects an `ImageChunk` whose `expected_tokens` don't match. The live probe matched these numbers.
- **Qwen3.6 image tokens:** about (W/32)×(H/32). A 512×384 image is 192 tokens; the mean on this set is 173.
- **Inkling image tokens:** a 512×384 image is 130 tokens; the mean is 118.
- **Prompt totals:** FT prompt averages 203 tokens. Qwen zero-shot (with class list) averages 352. Inkling zero-shot averages 292.
- **Output tokens:** about 4 for Qwen zero-shot, about 6 for Inkling at effort 0.0 (measured), and an assumed 8 for FT.

| item | tokens | $ |
|---|---|---|
| Train, 3 epochs × 415,913 tokens | 1.25M | **1.47** |
| In-training val accuracy, 6 × 150 images | 6 × 30.6k | 0.11 |
| Eval base Qwen zero-shot, 300 | 105.7k in | 0.06 |
| Eval FT Qwen, 300 | 61.0k in | 0.04 |
| Eval Inkling zero-shot, 100 (300 → $0.17) | 29.2k in | 0.06 |
| Checkpoint storage (rough) | – | 0.05 |
| Probe (done) | – | 0.004 |
| **Projected total** | | **≈ $1.78** of the $10 budget |

Serving cost per 1,000 photos, before any prefix-cache discount:

| model | $ per 1,000 photos |
|---|---|
| FT Qwen | **$0.12** |
| Base Qwen zero-shot | $0.20 |
| Inkling zero-shot, effort 0 | $0.58 (about 4.8× FT) |

That leaves room for a second run, for example a rank or learning-rate ablation (about $1.5 each), or an Inkling column at higher effort.

## Results and serving (Oct 6)
For the full results see `results/RESULTS.md` and `results/chart.png`.
- **Accuracy (300 test images):** the fine-tuned model scored 79.7% [75.0, 84.0], versus 67.7% for base Qwen zero-shot and 49.7% for Inkling zero-shot.
- **Cost per 1,000 photos:** $0.10 vs $0.12 vs $0.33.
- **Serving:** use `serving/fieldsight_infer.py` with the native Tinker SDK and model path
  `tinker://f5090c0f-f54a-53c7-8aa7-e6486f065aef:train:0/sampler_weights/final`.
  Tinker's OpenAI-compatible endpoint rejects image input for this model.
