# Design: FieldSight Garden web app

## Context

`serving/fieldsight_infer.py` rebuilds the training prompt and calls Tinker's native `SamplingClient`. It imports `tinker` at import time and loads the Qwen tokenizer inside `FieldSight()`. The OpenAI-compatible endpoint returns HTTP 400 for images on this checkpoint, so the web process has to use that native client. Render Starter has 512 MB of RAM. The service must boot there without torch.

## Goals / Non-Goals

- One photo in, plant, condition, and three care steps out, read aloud in the browser.
- Run locally and in CI with no API key.
- Deploy from `render.yaml` on the Python starter plan.
- Keep photos out of storage. Optional daily counts only.
- Leave the sampler prompt, the LoRA, and the fix wording untouched.

## Decisions

1. **Wrap the client; do not reimplement it.** Real diagnoses call `FieldSight.diagnose`. The web layer adds plant and condition names from `fieldsight/labels.py`, passes confidence through only when the client supplies it, and attaches the fix payload the client already returns.
2. **Mock mode never imports `tinker` or `transformers`.** `FIELDSIGHT_MOCK=1` picks a real class from a hash of the image bytes and reads `care/fixes.json`. Tests and CI use this path.
3. **`GET /health` is liveness.** It returns 200 as soon as the process can serve, with `ready` true or false. The model loads on a background thread because the tokenizer import and the sampling client are slow. `POST /api/diagnose` returns 503 until that load finishes, or until it fails. Render's health check can pass while the model wakes up.
4. **Limits live in the process.** Each IP gets a small per-minute budget, and the whole process has a global budget, so a public URL cannot drain the Tinker key unchecked. Uploads are capped (default 8 MB) and opened with Pillow before any model call. Pillow's pixel cap is lowered so a huge image fails closed.
5. **Stats are optional.** `DATABASE_URL` selects Postgres and a `diagnoses` table of label, latency, and timestamp. No image column exists. If the variable is unset, or the database cannot be reached, counts stay in memory for this process and diagnosis still succeeds.
6. **Speech is browser-side.** `SpeechSynthesis` runs from the tap that asks to hear the steps. The server does not synthesize audio.
7. **No Dockerfile.** `tinker` does not depend on torch unless the `torch` extra is installed. The native Python runtime plus `requirements.txt` is enough.
8. **The blueprint prefetches the tokenizer.** The build downloads the Qwen tokenizer into `HF_HOME` inside the slug so the first request does not wait on Hugging Face.

## Risks / Trade-offs

- Importing `transformers` on a 512 MB instance can leave little free RAM. A larger plan is the escape hatch if the process is killed while loading.
- The checkpoint is private. The API key has to belong to an account that can sample `FIELDSIGHT_MODEL_PATH`.
- In-memory counts reset on restart and are not shared across instances. Postgres is what makes "diagnoses today" durable.
- Render's `basic-256mb` Postgres is a paid plan. Deleting the database block leaves a working app.
- The client IP is the last `X-Forwarded-For` hop, which matches a single Render proxy. A different edge setup would need a different rule.
- Mock labels are a sample from the fix table. The page says so when mock mode is on.

## Migration Plan

- First deploy: set `TINKER_API_KEY` in the Render dashboard (`sync: false` prompts for it). Confirm `GET /health` then a real photo.
- To run without a database, remove the `databases` entry and the `DATABASE_URL` variable before applying the blueprint.
- Rollback is reverting the service deploy. The checkpoint and the fix table are unchanged.

## Open Questions

- None for Week 1. Confidence stays empty until the sampler returns one; the card shows it only then.
