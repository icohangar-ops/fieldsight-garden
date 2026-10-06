#!/usr/bin/env python
"""LoRA SFT of Qwen3.6-35B-A3B on the FieldSight PlantDoc split, via tinker-cookbook's
supervised training loop (same machinery as tinker_cookbook.recipes.vlm_classifier.train,
but with our local data, short prompt and single-class-string target).

Defaults (conservative, sized to a $10 total budget):
  rank 32, LR 5e-4 (= tinker_cookbook.hyperparam_utils.get_lr for this model), cosine schedule,
  batch 32, 3 epochs (~190 steps), hflip augmentation p=0.5, val accuracy every 32 steps
  on the 150-image val split (sampling, cheap), checkpoints every 64 steps with a 7-day TTL.
  Projected train spend ~$1.47 for 3 epochs (415,913 train tokens/epoch x $1.177/M).

Usage:
  python scripts/train.py log_dir=runs/ft-r32-lr5e-4            # full run
  python scripts/train.py log_dir=runs/smoke max_steps=2 eval_every=0 save_every=0   # ~$0.02 smoke test
  python scripts/train.py log_dir=runs/x dry_run=True             # cost estimate only, no API calls
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sys
from pathlib import Path

import chz
import tinker
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fieldsight.common import (  # noqa: E402
    QWEN, QWEN_RENDERER, build_generation_prompt, build_supervised_example, get_renderer,
    image_path, load_image, load_split, parse_label, parse_response_text, train_cost,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator  # noqa: E402
from tinker_cookbook.supervised import train as sl_train  # noqa: E402
from tinker_cookbook.supervised.common import datum_from_model_input_weights  # noqa: E402
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder  # noqa: E402


class PlantDocDataset(SupervisedDataset):
    def __init__(self, split: str, model_name: str, batch_size: int, hflip_p: float, seed: int = 0):
        self.rows = load_split(split)
        self.model_name, self.batch_size, self.hflip_p = model_name, batch_size, hflip_p
        self._rng = random.Random(seed)
        self.set_epoch(seed)

    def set_epoch(self, seed: int = 0):
        self.order = list(range(len(self.rows)))
        random.Random(seed).shuffle(self.order)

    def __len__(self):
        return math.ceil(len(self.rows) / self.batch_size)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        out = []
        for i in self.order[index * self.batch_size:(index + 1) * self.batch_size]:
            r = self.rows[i]
            im = load_image(image_path(r))
            if self._rng.random() < self.hflip_p:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mi, w = build_supervised_example(self.model_name, im, r["label"])
            out.append(datum_from_model_input_weights(mi, w, max_length=4096))
        return out


@chz.chz
class PlantDocBuilder(SupervisedDatasetBuilder):
    model_name: str = QWEN
    batch_size: int = 32
    hflip_p: float = 0.5

    def __call__(self):
        return PlantDocDataset("train", self.model_name, self.batch_size, self.hflip_p), None


class ValAccuracy(SamplingClientEvaluator):
    def __init__(self, model_name: str, n: int | None):
        rows = load_split("val")
        self.rows = rows[:n] if n else rows
        self.model_name = model_name

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        params = tinker.SamplingParams(max_tokens=16, temperature=0.0,
                                       stop=get_renderer(self.model_name).get_stop_sequences())
        sem = asyncio.Semaphore(32)

        async def one(r):
            async with sem:
                p = build_generation_prompt(self.model_name, load_image(image_path(r)), "ft")
                resp = await sampling_client.sample_async(prompt=p, num_samples=1, sampling_params=params)
                strict, lenient = parse_label(parse_response_text(self.model_name, list(resp.sequences[0].tokens)))
                return strict == r["label"], lenient == r["label"]

        res = await asyncio.gather(*[one(r) for r in self.rows])
        return {"val/acc_strict": sum(a for a, _ in res) / len(res),
                "val/acc_lenient": sum(b for _, b in res) / len(res)}


@chz.chz
class ValAccuracyBuilder:
    model_name: str = QWEN
    n: int | None = None

    def __call__(self) -> ValAccuracy:
        return ValAccuracy(self.model_name, self.n)


@chz.chz
class Args:
    log_dir: str = "runs/ft-qwen36-r32"
    model_name: str = QWEN
    lora_rank: int = 32
    learning_rate: float = 5e-4
    lr_schedule: str = "cosine"
    num_epochs: int = 3
    batch_size: int = 32
    hflip_p: float = 0.5
    eval_every: int = 32
    val_n: int | None = None  # None = all 150 val images
    save_every: int = 64
    ttl_seconds: int = 7 * 24 * 3600
    max_steps: int | None = None
    max_usd: float = 3.0  # abort before starting if projected train spend exceeds this
    load_checkpoint_path: str | None = None
    wandb_project: str | None = None
    behavior_if_log_dir_exists: str = "ask"
    dry_run: bool = False


def projected_train_usd(a: Args) -> tuple[float, int]:
    # exact token count of one epoch from the offline accounting (falls back to a fresh count)
    cp = ROOT / "results/cost_probe.json"
    if cp.exists():
        tok_epoch = json.loads(cp.read_text())["offline"]["train"]["train_datum_tokens_sum"]
    else:
        ds = PlantDocDataset("train", a.model_name, 10_000, 0.0)
        tok_epoch = sum(d.model_input.length for d in ds.get_batch(0))
    n_rows = len(load_split("train"))
    steps_per_epoch = math.ceil(n_rows / a.batch_size)
    total_steps = steps_per_epoch * a.num_epochs
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    tokens = tok_epoch * total_steps / steps_per_epoch
    return train_cost(a.model_name, int(tokens)), total_steps


def main(a: Args):
    usd, steps = projected_train_usd(a)
    print(f"Projected train spend: ${usd:.3f} for {steps} steps "
          f"({a.num_epochs} epochs x batch {a.batch_size}, rank {a.lora_rank}, LR {a.learning_rate})")
    if usd > a.max_usd:
        raise SystemExit(f"Projected ${usd:.2f} > max_usd ${a.max_usd}; raise max_usd to proceed.")
    if a.dry_run:
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set")
    cfg = sl_train.Config(
        log_path=str(ROOT / a.log_dir),
        model_name=a.model_name,
        recipe_name="fieldsight_garden_plantdoc",
        renderer_name=QWEN_RENDERER,
        load_checkpoint_path=a.load_checkpoint_path,
        dataset_builder=PlantDocBuilder(model_name=a.model_name, batch_size=a.batch_size, hflip_p=a.hflip_p),
        evaluator_builders=[ValAccuracyBuilder(model_name=a.model_name, n=a.val_n)] if a.eval_every else [],
        learning_rate=a.learning_rate,
        lr_schedule=a.lr_schedule,
        num_epochs=a.num_epochs,
        lora_rank=a.lora_rank,
        save_every=a.save_every,
        eval_every=a.eval_every,
        ttl_seconds=a.ttl_seconds,
        max_steps=a.max_steps,
        wandb_project=a.wandb_project,
    )
    from tinker_cookbook import cli_utils
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=a.behavior_if_log_dir_exists)
    asyncio.run(sl_train.main(cfg))
    print(f"Done. Final sampler path is in {cfg.log_path}/checkpoints.jsonl")


if __name__ == "__main__":
    main(chz.entrypoint(Args))
