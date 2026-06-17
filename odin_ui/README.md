# Odin UI

A TypeScript (React + Vite) web UI for Odin. Upload a face photo and get it back
with the MediaPipe landmarks drawn on top, the calculated facial ratios listed
beside it, and the model's 1–10 attractiveness score.

It does **not** reimplement the analysis in the browser — a small FastAPI backend
reuses the exact Odin Python pipeline (landmarks → ratios → XGBoost score), so the
results always match the CLI/model.

```
odin_ui/
├── backend/    FastAPI wrapper around the Odin pipeline
└── frontend/   Vite + React + TypeScript app
```

## Run

**1. Backend** (uses the project's main venv, which has mediapipe/xgboost/etc.):

```bash
# from the repo root, activate the project venv, then:
cd odin_ui/backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The backend imports the `Odin` package from the repo root and loads
`models/model_{male,female}.joblib` + `models/face_landmarker.task`, so run it
with the same Python environment that runs the CLI.

**2. Frontend:**

```bash
cd odin_ui/frontend
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173). The dev server proxies
`/api/*` to the backend on port 8000.

## How it works
- `POST /api/analyze` (multipart: `file`, `sex`) → JSON with `width/height`,
  `score`, `score_raw`, `boosted`, `landmarks` (pixel coords), `ratios`,
  `appearance`, and `colors`.
- The frontend draws the photo to a canvas at its native resolution, overlays the
  478 landmark points, and renders the ratios/score panel. The male score
  includes the presentation boost (`score_raw` is the pre-boost value).
