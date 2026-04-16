"""
main.py
-------
Application entry point for the Athlix AI Injury Prediction backend.

Initialises the FastAPI application, registers routers, configures CORS,
and wires up lifecycle events for resource management.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models.schemas import HealthResponse
from app.routes.upload_route import router as upload_router

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_VERSION = "0.1.0"
APP_TITLE   = "Athlix — AI Injury Prediction API"
APP_DESC    = """
## Overview
Backend service for the **Athlix** AI-powered sports injury prediction system.

### Capabilities (v0.1)
- Upload a **single image frame** → receive detected pose landmarks and joint angles.
- Upload a **video file**         → receive per-frame landmarks and biomechanical features.

### Upcoming (v0.2)
- ML model inference for real injury-risk scoring.
- WebSocket streaming for real-time pose analysis.
- Athlete session history and longitudinal risk tracking.
"""

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESC,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    # Adjust `allow_origins` for production to restrict to your frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # Replace with specific origins in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(upload_router)

    # ── Startup / shutdown events ────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("Athlix API v%s starting up.", APP_VERSION)
        # Future: pre-load ML model here to avoid cold-start latency on first request.

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("Athlix API shutting down — cleaning up resources.")

    # ── Health check ─────────────────────────────────────────────────────────
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="API health check",
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=APP_VERSION,
            ml_model_loaded=False,   # Update to True once model.pkl is loaded
        )

    # ── Root ─────────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "message": "Athlix AI Injury Prediction API",
                "version": APP_VERSION,
                "docs": "/docs",
            }
        )

    return app


# ---------------------------------------------------------------------------
# ASGI application instance (used by uvicorn)
# ---------------------------------------------------------------------------
app = create_app()
