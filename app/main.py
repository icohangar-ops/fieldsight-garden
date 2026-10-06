"""FieldSight Garden HTTP service."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from app.diagnose import (
    BadImage,
    MockEngine,
    ModelNotReady,
    RealEngine,
    TooLarge,
    read_limited,
    validate_image,
)
from app.limit import RateLimiter
from app.settings import ROOT, Settings
from app.stats import open_store, today_utc

logger = logging.getLogger("fieldsight")

STATIC = Path(__file__).resolve().parent / "static"
INDEX = STATIC / "index.html"
CHART = ROOT / "results" / "chart.png"


def _size_phrase(n: int) -> str:
    if n >= 1024 * 1024 and n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)} MB"
    if n >= 1024:
        return f"{max(n // 1024, 1)} KB"
    return f"{n} bytes"


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        parts = [part.strip() for part in forwarded.split(",") if part.strip()]
        if parts:
            return parts[-1]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine: MockEngine | RealEngine = MockEngine() if settings.mock else RealEngine(settings.model_path)
    store = open_store(settings.database_url)
    per_ip = RateLimiter(settings.rate_limit, settings.rate_window_s)
    global_limit = RateLimiter(settings.global_rate_limit, settings.rate_window_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.load_model and not settings.mock:
            engine.start()
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="FieldSight Garden", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.store = store

    @app.middleware("http")
    async def api_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api") or request.url.path == "/health":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "mock": settings.mock,
            "ready": bool(engine.ready),
            "failed": bool(engine.failed),
            "storage": store.kind,
            "storage_fallback": bool(store.fallback),
            "model_path": settings.model_path,
            "detail": engine.detail,
        }

    @app.get("/api/stats")
    def stats() -> dict:
        try:
            count = store.count_today()
        except Exception:
            logger.exception("failed to read diagnosis count")
            count = 0
        return {
            "diagnoses_today": count,
            "storage": store.kind,
            "day": today_utc(),
        }

    @app.post("/api/diagnose")
    async def diagnose(request: Request, image: UploadFile | None = File(default=None)) -> dict:
        retry = {"Retry-After": str(settings.rate_window_s)}
        if not per_ip.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Too many photos from this phone. Wait a minute and try one leaf.", headers=retry)
        if not global_limit.allow("global"):
            raise HTTPException(status_code=429, detail="FieldSight is busy. Wait a minute and try one leaf.", headers=retry)
        if image is None:
            raise HTTPException(status_code=400, detail="Choose a leaf photo.")
        try:
            data = await read_limited(image, settings.max_bytes)
            validate_image(data, max_bytes=settings.max_bytes)
            result = engine.diagnose(data, max_bytes=settings.max_bytes)
        except TooLarge:
            raise HTTPException(status_code=413, detail=f"Photo is too large. Send one leaf under {_size_phrase(settings.max_bytes)}.")
        except BadImage:
            raise HTTPException(status_code=400, detail="That file isn't a photo I can read. Try a JPEG or PNG of one leaf.")
        except ModelNotReady as exc:
            raise HTTPException(status_code=503, detail=exc.detail)
        except Exception:
            logger.exception("diagnosis failed")
            raise HTTPException(status_code=502, detail="The garden model didn't answer. Try another photo in a moment.")

        try:
            store.record(result.get("label"), result.get("latency_s"))
        except Exception:
            logger.exception("failed to record diagnosis count")
        try:
            result["diagnoses_today"] = store.count_today()
        except Exception:
            logger.exception("failed to read diagnosis count")
            result["diagnoses_today"] = None
        logger.info(
            "diagnosis label=%s latency_s=%s mock=%s bytes=%s",
            result.get("label"),
            result.get("latency_s"),
            result.get("mock"),
            len(data),
        )
        return result

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(INDEX, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})

    @app.get("/chart.png")
    def chart() -> FileResponse:
        return FileResponse(CHART, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    return app


app = create_app()
