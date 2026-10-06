# Tasks

## 1. Service

- [x] 1.1 Add FastAPI routes: `GET /health`, `POST /api/diagnose`, `GET /api/stats`, the page, and the results chart.
- [x] 1.2 Add mock mode that uses `care/fixes.json` and does not import `tinker`.
- [x] 1.3 Add upload size limits, image checks, and per-IP plus global rate limits.
- [x] 1.4 Store label, latency, and timestamp in Postgres when `DATABASE_URL` is set, and in memory otherwise.

## 2. Page

- [x] 2.1 Build the mobile-first page with rear-camera capture, a result card, one-tap speech, and a finish state.
- [x] 2.2 Show the results chart, the accuracy and cost table, PlantDoc CC BY 4.0 attribution, and the extension-source disclaimer.

## 3. Deploy and docs

- [x] 3.1 Add `render.yaml` for a Python starter web service and optional Postgres.
- [x] 3.2 Add light runtime requirements (no torch) and mock-mode API tests.
- [x] 3.3 Update `README.md` with run steps, the results table, and a Deploy to Render button.

## 4. Done

- [x] 4.1 Run the app in mock mode, pass tests, and validate `render.yaml` against the Render schema.
