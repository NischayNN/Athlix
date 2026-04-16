from fastapi import APIRouter, File, UploadFile, status, HTTPException
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

# removed unused RiskAssessmentRequest
from app.services.pipeline import run_pipeline

router = APIRouter(prefix="", tags=["Pipeline"])

@router.post("/analyze", summary="Full Video Analysis Pipeline")
async def analyze_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    # For now, we will simulate the extraction of form_decay from the video
    # and provide realistic default values for the other coaching inputs
    # since the frontend Upload form only provides a video file.
    
    # Let's say we use the pipeline with some inferred or mocked features
    # that represent a typical athlete video upload scenario.
    
    # In a full integration, you would process the video with PoseService,
    # extract the max/average form_deviation_score, and then feed it here.
    
    # Since we are trying to just get the 'Upload video -> API -> result display'
    # working as requested, we'll run the unified pipeline.
    
    input_features = {
        "training_load": 7.5,
        "recovery_score": 45.0,
        "fatigue_index": 6.0,
        "form_decay": 0.65, # Simulated moderate form decay from video
        "previous_injury": 0
    }
    
    result = run_pipeline(input_features)
    return result.to_dict()
