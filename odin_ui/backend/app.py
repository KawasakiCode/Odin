"""
Odin API — a thin FastAPI wrapper around the existing Odin Python pipeline.

POST /analyze  (multipart: file=<image>, sex=male|female)
  -> { width, height, sex, score, score_raw, landmarks[[x,y]...],
       ratios[...], appearance[...], colors[...] }

The landmarks, ratios and score are produced by the SAME code the CLI uses
(Odin.Face_analysis.* + Odin.main), so the UI never diverges from the model.

Run (from this folder, using the project venv that has the pipeline deps):
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
"""
import math
import os
import sys
import tempfile
from pathlib import Path

# Repo root = three levels up: odin_ui/backend/app.py -> repo root.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.trichion import apply_trichion
from Odin.Face_analysis.Ratios.appearance import appearance_features
from Odin.main import add_shape_features, build_features, feature_contributions, male_boost, MODEL_PATHS

app = FastAPI(title="Odin API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Load both model bundles once at startup.
MODELS = {sex: joblib.load(path) for sex, path in MODEL_PATHS.items()}

# Human-readable labels for the 40 model features.
LABELS = {
    "symmetry": "Facial symmetry (lower = better)",
    "width_ratio_46": "Eye spacing / cheekbone (46% ideal)",
    "height_ratio_36": "Midface height (36% ideal)",
    "canthal_average": "Canthal tilt (°)",
    "fwhr": "Facial width-to-height",
    "jaw_contour_deviation": "Jaw contour deviation (°)",
    "jaw_contour_jaw_slope": "Jaw slope (°)",
    "jaw_contour_canthus_alare_slope": "Canthus–alare slope (°)",
    "upper_third": "Upper third %",
    "middle_third": "Middle third %",
    "lower_third": "Lower third %",
    "bizygomatic_bigonial_ratio": "Cheekbone / jaw taper",
    "facial_fifths": "Facial fifths",
    "inter_eye_ratio": "Inter-eye / eye width",
    "orbitonasal_ratio": "Nose width / eye spacing",
    "nasofacial_proportion": "Nose width / face",
    "naso_oral_ratio": "Mouth / nose width",
    "face_golden_ratio": "Face height / width (golden)",
    "face_height_bigonial_width": "Face height / jaw width",
    "lip_vermilion_ratio": "Lower / upper lip",
    "lower_third_upper_split": "Lower third — upper %",
    "lower_third_lower_split": "Lower third — lower %",
    "stomion_canthus_ratio": "Stomion–menton / canthus",
    "ear_avg": "Eye aspect ratio (openness)",
    "ear_asymmetry": "Eye asymmetry",
    "skin_texture": "Skin texture (Laplacian var.)",
    "eye_skin_luminance_contrast": "Eye–skin luminance contrast",
    "lip_skin_luminance_contrast": "Lip–skin luminance contrast",
    "lip_skin_redness_contrast": "Lip–skin redness contrast",
    "eye_skin_redness_contrast": "Eye–skin redness contrast",
    "facial_contrast_avg": "Facial contrast (avg)",
}

# Geometric ratios vs colour/texture appearance features (raw RGB channels are
# omitted from the lists — they're surfaced as colour swatches instead).
GEOM_KEYS = [
    "symmetry", "width_ratio_46", "height_ratio_36", "canthal_average", "fwhr",
    "jaw_contour_deviation", "jaw_contour_jaw_slope",
    "jaw_contour_canthus_alare_slope", "upper_third", "middle_third",
    "lower_third", "bizygomatic_bigonial_ratio", "facial_fifths",
    "inter_eye_ratio", "orbitonasal_ratio", "nasofacial_proportion",
    "naso_oral_ratio", "face_golden_ratio", "face_height_bigonial_width",
    "lip_vermilion_ratio", "lower_third_upper_split", "lower_third_lower_split",
    "stomion_canthus_ratio", "ear_avg", "ear_asymmetry",
]
APPEARANCE_KEYS = [
    "skin_texture", "eye_skin_luminance_contrast", "lip_skin_luminance_contrast",
    "lip_skin_redness_contrast", "eye_skin_redness_contrast", "facial_contrast_avg",
]

# MediaPipe 478-mesh indices each geometric ratio depends on (mirrors the
# face_data.py mapping). The UI highlights only these points when a ratio is
# hovered, and the union of all of them is the default overlay. Composite points
# expand to their source indices (trichion -> 10/8/9, glabella -> 8/9, etc.).
RATIO_LANDMARKS = {
    "symmetry": [33, 133, 263, 362, 159, 145, 386, 374, 70, 107, 46, 55, 52, 105,
                 300, 336, 276, 285, 282, 334, 48, 278, 61, 291, 234, 454, 132,
                 361, 58, 288, 50, 280, 205, 425, 10, 8, 9, 168, 4, 2, 0, 152],
    "width_ratio_46": [468, 473, 234, 454],
    "height_ratio_36": [468, 473, 13, 14, 10, 8, 9, 152],
    "canthal_average": [33, 133, 263, 362],
    "fwhr": [234, 454, 55, 285, 46, 276, 0],
    "jaw_contour_deviation": [33, 48, 132, 152, 263, 278, 361],
    "jaw_contour_jaw_slope": [132, 152, 361],
    "jaw_contour_canthus_alare_slope": [33, 48, 263, 278],
    "upper_third": [10, 8, 9, 2, 152],
    "middle_third": [10, 8, 9, 2, 152],
    "lower_third": [10, 8, 9, 2, 152],
    "bizygomatic_bigonial_ratio": [234, 454, 58, 288],
    "facial_fifths": [33, 133, 263, 362, 234, 454],
    "inter_eye_ratio": [33, 133, 263, 362],
    "orbitonasal_ratio": [48, 278, 133, 362],
    "nasofacial_proportion": [48, 278, 234, 454],
    "naso_oral_ratio": [61, 291, 48, 278],
    "face_golden_ratio": [10, 8, 9, 152, 234, 454],
    "face_height_bigonial_width": [10, 8, 9, 152, 132, 361],
    "lip_vermilion_ratio": [0, 13, 14, 17],
    "lower_third_upper_split": [2, 13, 14, 152],
    "lower_third_lower_split": [2, 13, 14, 152],
    "stomion_canthus_ratio": [13, 14, 152, 33, 263],
    "ear_avg": [159, 145, 33, 133, 386, 374, 263, 362],
    "ear_asymmetry": [159, 145, 33, 133, 386, 374, 263, 362],
}

# Ideal/target per ratio as (male, female) display strings (ranges kept as text).
# Sources: the docstrings in calculate_ratios.py.
IDEALS = {
    "symmetry": ("→ 0", "→ 0"),
    "width_ratio_46": ("0.46", "0.46"),
    "height_ratio_36": ("0.36", "0.36"),
    "canthal_average": ("+3 to +5°", "+5 to +8°"),
    "fwhr": ("1.90–2.05", "1.75–1.90"),
    "jaw_contour_deviation": ("0–15°", "0–15°"),
    "jaw_contour_jaw_slope": ("—", "—"),
    "jaw_contour_canthus_alare_slope": ("—", "—"),
    "upper_third": ("0.310", "0.295"),
    "middle_third": ("0.305", "0.324"),
    "lower_third": ("0.385", "0.382"),
    "bizygomatic_bigonial_ratio": ("1.128", "1.174"),
    "facial_fifths": ("5.00", "5.00"),
    "inter_eye_ratio": ("0.95–1.15", "0.95–1.15"),
    "orbitonasal_ratio": ("1.00", "≤1.00"),
    "nasofacial_proportion": ("0.25", "0.25"),
    "naso_oral_ratio": ("1.50–1.62", "1.50–1.62"),
    "face_golden_ratio": ("1.35", "1.30"),
    "face_height_bigonial_width": ("1.566", "1.613"),
    "lip_vermilion_ratio": ("1.60", "1.60"),
    "lower_third_upper_split": ("0.30", "0.30"),
    "lower_third_lower_split": ("0.70", "0.70"),
    "stomion_canthus_ratio": ("0.618", "0.618"),
    "ear_avg": ("0.20–0.25", "0.30–0.35"),
    "ear_asymmetry": ("→ 0", "→ 0"),
}


def _num(v):
    """JSON-safe number: floats only, NaN/inf -> None."""
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else round(v, 4)


def _items(feats, keys, sex):
    idx = 0 if sex == "male" else 1
    return [{
        "key": k,
        "label": LABELS.get(k, k),
        "value": _num(feats.get(k)),
        "ideal": IDEALS[k][idx] if k in IDEALS else None,
        "landmarks": RATIO_LANDMARKS.get(k, []),
    } for k in keys]


def _hex(feats, prefix):
    r, g, b = feats.get(f"{prefix}_r"), feats.get(f"{prefix}_g"), feats.get(f"{prefix}_b")
    if r is None or g is None or b is None:
        return None
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, int(round(r)))),
        max(0, min(255, int(round(g)))),
        max(0, min(255, int(round(b)))),
    )


@app.get("/health")
def health():
    return {"ok": True, "models": list(MODELS)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), sex: str = Form("male")):
    sex = (sex or "").lower()
    if sex not in MODELS:
        raise HTTPException(400, "sex must be 'male' or 'female'")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")

    # Reuse calculate_landmarks_array (which reads from a path) by writing the
    # upload to a temp file, so the pipeline stays byte-for-byte identical.
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        try:
            landmarks, img = calculate_landmarks_array(tmp.name)
        except (ValueError, FileNotFoundError) as e:
            # No face / unreadable image -> 422 with the pipeline's message.
            raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp.name)

    h, w = img.shape[:2]
    px = landmarks.copy()
    px[:, 0] *= w
    px[:, 1] *= h
    px[:, 2] *= w

    face_data = extract_face_data(px)
    # Detect the real hairline and overwrite the trichion BEFORE the ratios, so
    # the UI's 4 trichion-dependent ratios match the CLI and training exactly.
    trichion_pt, _ = apply_trichion(face_data, img)
    feats = build_features(face_data, appearance_features(img, px))

    model = MODELS[sex]
    add_shape_features(feats, px, model)
    X = pd.DataFrame([feats], columns=model["feature_names"])
    raw = float(model["xgboost"].predict(X)[0])
    base, contribs = feature_contributions(model, X)
    score = male_boost(raw, model) if model.get("label") == "MALE" else raw

    contrib_items = sorted(
        ({"key": k,
        "label": LABELS.get(k, k),
        "value": _num(feats.get(k)),
        "contribution": round(float(c), 3),
        "landmarks": RATIO_LANDMARKS.get(k, [])}
        for k, c in contribs.items()),
        key=lambda d: abs(d["contribution"]), reverse=True,
    )

    return {
        "width": w,
        "height": h,
        "sex": sex,
        "score": round(score, 2),
        "score_raw": round(raw, 2),
        "boosted": model.get("label") == "MALE" and abs(score - raw) > 1e-6,
        "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y, _ in px],
        "trichion": ([round(float(trichion_pt[0]), 1), round(float(trichion_pt[1]), 1)]
                     if trichion_pt is not None else None),
        "ratios": _items(feats, GEOM_KEYS, sex),
        "appearance": _items(feats, APPEARANCE_KEYS, sex),
        "contribs": contrib_items,
        "base": round(float(base), 3),
        "colors": [
            {"label": "Lips", "hex": _hex(feats, "lips")},
            {"label": "Eyes", "hex": _hex(feats, "eye")},
            {"label": "Skin", "hex": _hex(feats, "skin")},
        ],
    }
