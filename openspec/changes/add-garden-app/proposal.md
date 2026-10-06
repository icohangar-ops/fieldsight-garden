# Proposal: FieldSight Garden web app

## Why

The fine-tuned model, the fix table, and the native sampling client already exist. A gardener still has no way to snap one leaf, hear the three steps, and put the phone down. Week 1 needs that app on a small Render instance.

## What Changes

- A FastAPI service wraps `serving/fieldsight_infer.py`: `POST` an image, get JSON back, and `GET /health`.
- Mock mode (`FIELDSIGHT_MOCK=1`) answers from `care/fixes.json` without a Tinker key, torch, or a tokenizer download.
- A mobile-first page: rear-camera capture, a result card, one-tap `SpeechSynthesis`, and a "Done, go outside" finish state.
- Anonymous diagnosis counts (label, latency, timestamp) go to Postgres when `DATABASE_URL` is set. The app still runs with no database.
- A Render blueprint for a Python starter web service and an optional small Postgres instance.
- API tests in mock mode, plus run and deploy steps in the README.

## Capabilities

### New Capabilities

- `garden-app`: public leaf diagnosis, health, limits, optional daily counts, and the single page that reads the fix aloud.

### Modified Capabilities

- None. Training, eval, and the token sequence in `serving/fieldsight_infer.py` stay as they are.

## Impact

- New application code under `app/`, tests under `tests/`, `requirements.txt`, `requirements-dev.txt`, and `render.yaml`.
- `README.md` gains run, test, and deploy steps, the accuracy and cost table, and a Deploy to Render button.
- `TINKER_API_KEY` stays in the environment. It is not written into the repo.
- Each real photo calls the private Tinker checkpoint and spends a fraction of a cent.
