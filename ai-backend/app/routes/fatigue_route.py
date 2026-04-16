from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import FatigueInput, FatigueResult
from app.services.feature_engineering import compute_fatigue_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Fatigue & Load"])


@router.post("/fatigue", response_model=FatigueResult, summary="Compute fatigue and recovery metrics")
async def fatigue_endpoint(payload: FatigueInput) -> FatigueResult:
    try:
        return compute_fatigue_metrics(payload)
    except Exception as exc:
        logger.exception("Fatigue computation failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
