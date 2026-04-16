from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models.schemas import HealthResponse
from app.routes.upload_route   import router as upload_router
from app.routes.pose_route     import router as pose_router
from app.routes.fatigue_route  import router as fatigue_router
from app.routes.process_route  import router as process_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

APP_VERSION = "0.2.0"
APP_TITLE   = "Athlix — AI Injury Prediction API"


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(pose_router)
    app.include_router(upload_router)
    app.include_router(fatigue_router)
    app.include_router(process_router)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("Athlix API v%s starting up.", APP_VERSION)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("Athlix API shutting down.")

    @app.get("/health", response_model=HealthResponse, tags=["System"], summary="API health check")
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=APP_VERSION, ml_model_loaded=False)

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(content={"message": "Athlix AI Injury Prediction API", "version": APP_VERSION, "docs": "/docs"})

    return app


app = create_app()
