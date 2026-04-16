# Athlix — AI Injury Prediction System

Athlix is an advanced AI-powered biomechanical analysis system designed to predict and prevent athletic injuries. The application evaluates an athlete's physical form during structural exercises (like squats) or dynamic movements (like bowling), utilizing a powerful synergy of MediaPipe computer vision, custom biomechanics heuristics, and XGBoost machine learning to produce interactive, immediate, and actionable risk assessments.

## 🏗️ Architecture

The system is decoupled into an ML-heavy API backend and a lightweight, responsive React frontend.

- **`ai-backend/`**: A stateless **FastAPI** Python application. It handles image parsing safely, calls MediaPipe for anatomical point tracking, runs heuristic validation against physical laws, and funnels data through structured ML models (XGBoost/SciKitLearn) infused with global explainability metrics (SHAP).
- **`frontend/`**: A sleek **React + Vite** single-page application built with modern web aesthetics in mind (Tailwind CSS). This serves as the athlete or coach's dashboard to upload media, view generated coaching points, and observe biomechanical risk summaries.

---

## 🚀 Quick Setup Guide

### 1. Backend Installation (FastAPI)
The backend expects Python 3.10+ and houses heavy computational packages alongside its web server logic.

```bash
# 1. Provide the model dependencies
pip install -r requirements.txt

# 2. Navigate to the backend directory
cd ai-backend

# 3. CRITICAL: Train and serialize the initial Baseline ML models
python app/services/train_models.py

# 4. Start the ASGI Production Server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*The API should now be routing logic actively on `http://localhost:8000/docs`.*

### 2. Frontend Installation (React)
Open a separate secondary terminal root.

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install all Node Modules
npm install

# 3. Spin up the Vite Dev Server
npm run dev
```
*The application interface will gracefully load on `http://localhost:5173/`.*

---

## 🎯 Usage

1. Open your browser to `http://localhost:5173/`.
2. Click **Start Upload** to navigate to the analysis module.
3. Select an evaluation context (e.g. "Squat") for correct heuristic processing.
4. Drag-and-drop or select an `.mp4` video (or supported image frame) of the athletic movement. Ensure the movement hits the fundamental rules (direct side profile, fully illuminated, solitary isolation).
5. Press **Analyze**. The payload is sent down a custom multipart pipeline to the `http://localhost:8000/analyze` route.
6. The Backend evaluates fatigue, tracks multi-joint symmetry, grades deviations mathematically using regression modeling, evaluates risk layers, and returns a detailed `PipelineResult`.
7. Enjoy your fully documented **Coaching Report** alongside your computed relative **Risk Level**.
