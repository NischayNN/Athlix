# Athlix — AI Injury Prediction Backend

> FastAPI + MediaPipe BlazePose backend for real-time biomechanical analysis and injury risk prediction.

---

## Project Structure

```
ai-backend/
├── app/
│   ├── main.py                  # FastAPI app factory, CORS, health check
│   ├── routes/
│   │   └── upload_route.py      # POST /upload/frame  &  POST /upload/video
│   ├── services/
│   │   ├── pose_service.py      # MediaPipe BlazePose wrapper
│   │   └── feature_engineering.py  # Landmark → joint angles → feature vector
│   ├── utils/
│   │   └── angle_utils.py       # 3-D / 2-D vector angle math
│   └── models/
│       └── schemas.py           # Pydantic request / response schemas
├── data/
│   └── data.csv                 # Training dataset
├── model/
│   └── train_model.py           # XGBoost training script
└── requirements.txt
```

---

## Prerequisites

- Python **3.10** or **3.11**
- `pip` ≥ 23

---

## Quick Start

### 1 — Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Run the API server

```bash
# From inside ai-backend/
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000**.
Interactive docs: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Path             | Description                                       |
|--------|------------------|---------------------------------------------------|
| `GET`  | `/health`        | Returns API status and version.                   |
| `POST` | `/upload/frame`  | Upload a **JPEG / PNG image** for pose analysis.  |
| `POST` | `/upload/video`  | Upload an **MP4 / MOV video** for frame-by-frame analysis. |

### Example — Upload a frame (curl)

```bash
curl -X POST "http://localhost:8000/upload/frame" \
     -H "accept: application/json" \
     -F "file=@/path/to/image.jpg"
```

### Example response

```json
{
  "status": "success",
  "processing_time_ms": 142.5,
  "result": {
    "frame_index": 0,
    "pose_detected": true,
    "landmarks": [ ... ],
    "features": {
      "frame_index": 0,
      "joint_angles": {
        "left_knee": 165.3,
        "right_knee": 163.8,
        "left_hip": 172.1,
        ...
      },
      "symmetry_score": 0.987,
      "form_deviation_score": null,
      "predicted_risk_score": null
    }
  }
}
```

> **Note:** `predicted_risk_score` and `form_deviation_score` are `null` until the XGBoost ML model is integrated (see `model/train_model.py`).

---

## Training the ML Model

```bash
# From inside ai-backend/
python model/train_model.py
```

This creates `model/model.pkl`.  Wire it into `services/feature_engineering.py`'s `predict_risk()` function to enable live scoring.

---

## Architecture Overview

```
Upload (video / frame)
        │
        ▼
  upload_route.py          ← validates MIME type, manages temp files
        │
        ▼
  pose_service.py          ← BlazePose → List[Landmark]
        │
        ▼
  feature_engineering.py   ← Joint angles, symmetry, feature vector
        │
        ▼
  (ML model hook)          ← XGBoost → risk score  [coming soon]
        │
        ▼
  JSON Response
```

---

## Environment Variables (future)

| Variable              | Default | Description                        |
|-----------------------|---------|------------------------------------|
| `APP_ENV`             | `dev`   | `dev` / `prod`                     |
| `MAX_VIDEO_FRAMES`    | `300`   | Cap on frames processed per video  |
| `MODEL_PATH`          | `model/model.pkl` | Path to serialised ML model |

---

## Roadmap

- [x] MediaPipe BlazePose integration
- [x] Joint angle calculation (all major joints)
- [x] Modular service architecture
- [ ] XGBoost risk score inference
- [ ] WebSocket real-time streaming
- [ ] Athlete session history (PostgreSQL)
- [ ] Docker deployment
