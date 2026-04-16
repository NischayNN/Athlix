from fastapi import APIRouter, File, UploadFile, HTTPException
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

from app.services.pipeline import run_pipeline
from app.services.pose_service import analyze_video_form

router = APIRouter(prefix="", tags=["Pipeline"])


@router.post("/analyze", summary="Full Video Analysis Pipeline")
async def analyze_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    # ------------------------------------------------------------------ #
    # 1. Save the uploaded video to a temporary file so OpenCV can read it
    # ------------------------------------------------------------------ #
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file.close()
        tmp_path = tmp_file.name

        # ------------------------------------------------------------------ #
        # 2. Run real pose-based form analysis on the video (returns 0–100)
        # ------------------------------------------------------------------ #
        form_decay_score = analyze_video_form(tmp_path)          # 0–100

        # Normalise to 0–1 for the risk pipeline validator
        form_decay_normalized = round(form_decay_score / 100.0, 4)

        print(f"[DEBUG] form_decay (0-100) : {form_decay_score:.2f}")
        print(f"[DEBUG] form_decay (0-1)   : {form_decay_normalized:.4f}")

        # ------------------------------------------------------------------ #
        # 3. Build feature dict and run the full pipeline
        # ------------------------------------------------------------------ #
        input_features = {
            "training_load":  7.5,
            "recovery_score": 45.0,
            "fatigue_index":  6.0,
            "form_decay":     form_decay_normalized,
            "previous_injury": 0,
        }

        result = run_pipeline(input_features)
        return result.to_dict()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        # Always clean up the temp file
        Path(tmp_file.name).unlink(missing_ok=True)

