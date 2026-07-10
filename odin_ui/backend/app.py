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
    "skin_a_std": "Skin redness unevenness",
    "skin_b_std": "Skin tone unevenness",
    "skin_spot_burden": "Skin blemish / spot level",
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
    "skin_texture", "skin_a_std", "skin_b_std", "skin_spot_burden",
    "eye_skin_luminance_contrast", "lip_skin_luminance_contrast",
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

# Numeric ideal ranges per ratio, per sex, as (green_lo, green_hi, red_lo, red_hi):
# green = ideal band, yellow between green and red bounds, red beyond. Derived from
# the thresholds in calculate_ratios.py docstrings. Ratios in LOW_BETTER only have a
# high side (lower is better); their green_lo/red_lo are ignored. Ratios not listed
# (or with an "—" ideal) get no bar.
LOW_BETTER = {"symmetry", "ear_asymmetry", "jaw_contour_deviation"}
IDEAL_RANGES = {
    "symmetry":                    ((0, 0.035, 0, 0.09),)   * 2,
    "width_ratio_46":              ((0.44, 0.48, 0.40, 0.50),) * 2,
    "height_ratio_36":             ((0.34, 0.38, 0.33, 0.40),) * 2,
    "canthal_average":             ((3, 5, 0, 8), (5, 8, 2, 11)),
    "fwhr":                        ((1.90, 2.05, 1.70, 2.20), (1.75, 1.90, 1.60, 2.05)),
    "jaw_contour_deviation":       ((0, 15, 0, 25),) * 2,
    "upper_third":                 ((0.29, 0.33, 0.26, 0.36), (0.275, 0.315, 0.245, 0.345)),
    "middle_third":                ((0.285, 0.325, 0.255, 0.355), (0.304, 0.344, 0.274, 0.374)),
    "lower_third":                 ((0.36, 0.41, 0.32, 0.45), (0.36, 0.405, 0.32, 0.44)),
    "bizygomatic_bigonial_ratio":  ((1.05, 1.20, 1.00, 1.30), (1.10, 1.25, 1.00, 1.35)),
    "facial_fifths":               ((4.7, 5.3, 4.5, 5.5),) * 2,
    "inter_eye_ratio":             ((0.95, 1.15, 0.85, 1.25),) * 2,
    "orbitonasal_ratio":           ((0.90, 1.10, 0.80, 1.20), (0.85, 1.05, 0.75, 1.15)),
    "nasofacial_proportion":       ((0.23, 0.27, 0.20, 0.30),) * 2,
    "naso_oral_ratio":             ((1.50, 1.62, 1.30, 1.75),) * 2,
    "face_golden_ratio":           ((1.30, 1.40, 1.25, 1.45), (1.25, 1.35, 1.20, 1.40)),
    "face_height_bigonial_width":  ((1.48, 1.65, 1.40, 1.72), (1.55, 1.68, 1.45, 1.75)),
    "lip_vermilion_ratio":         ((1.40, 1.80, 1.20, 2.10),) * 2,
    "lower_third_upper_split":     ((0.25, 0.35, 0.20, 0.40),) * 2,
    "lower_third_lower_split":     ((0.65, 0.75, 0.60, 0.80),) * 2,
    "stomion_canthus_ratio":       ((0.56, 0.68, 0.50, 0.75),) * 2,
    "ear_avg":                     ((0.20, 0.25, 0.15, 0.32), (0.30, 0.35, 0.22, 0.42)),
    "ear_asymmetry":               ((0, 0.03, 0, 0.10),) * 2,
}


def _lerp_pos(v, xs, ys):
    """Piecewise-linear map of v through the (xs, ys) anchor points, clamped."""
    if v <= xs[0]:
        return ys[0]
    for i in range(len(xs) - 1):
        if v <= xs[i + 1]:
            d = xs[i + 1] - xs[i]
            t = (v - xs[i]) / d if d else 0.0
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _bar(key, value, sex):
    """Marker position (0..1 on a red-yellow-green-yellow-red bar) + status for a
    ratio value against its ideal range. None if the ratio has no numeric ideal."""
    rng = IDEAL_RANGES.get(key)
    if rng is None or value is None:
        return None
    glo, ghi, rlo, rhi = rng[0 if sex == "male" else 1]
    v = float(value)
    if key in LOW_BETTER:
        xs = [0.0, ghi, rhi, rhi + (rhi - ghi + 1e-6)]
        ys = [0.45, 0.58, 0.85, 0.97]
        status = "good" if v <= ghi else ("warn" if v <= rhi else "bad")
    else:
        pad = (rhi - rlo) * 0.6
        xs = [rlo - pad, rlo, glo, ghi, rhi, rhi + pad]
        ys = [0.03, 0.15, 0.35, 0.65, 0.85, 0.97]
        status = "good" if glo <= v <= ghi else ("warn" if rlo <= v <= rhi else "bad")
    return {"pos": round(_lerp_pos(v, xs, ys), 4), "status": status}


def ratio_lines(key, fd):
    """Measurement line segments (pixel coords) for a ratio, so the UI can draw
    what was actually computed on hover. Each segment is [[x1,y1],[x2,y2]]; mirrors
    the constructions documented in calculate_ratios.py. [] for non-geometric keys."""
    def P(k):
        v = fd[k]
        return [round(float(v[0]), 1), round(float(v[1]), 1)]

    def MID(a, b):
        va, vb = fd[a], fd[b]
        return [round((float(va[0]) + float(vb[0])) / 2, 1),
                round((float(va[1]) + float(vb[1])) / 2, 1)]

    st = MID("upper_lip_bottom_center", "lower_lip_top_center")   # stomion

    if key == "symmetry":                  # vertical axis of symmetry
        mids = ["top_center_forehead", "glabella", "top_of_nose_bridge", "nose_tip",
                "base_of_nose", "upper_lip_top_center", "chin"]
        mx = round(sum(float(fd[k][0]) for k in mids) / len(mids), 1)
        return [[[mx, round(float(fd["top_center_forehead"][1]), 1)],
                 [mx, round(float(fd["chin"][1]), 1)]]]
    if key == "width_ratio_46":            # IPD vs bizygomatic width
        return [[P("left_pupil_center"), P("right_pupil_center")],
                [P("left_zygomatic"), P("right_zygomatic")]]
    if key == "height_ratio_36":           # eye-to-mouth vs total face height
        return [[MID("left_pupil_center", "right_pupil_center"), st],
                [P("top_center_forehead"), P("chin")]]
    if key == "canthal_average":           # inner->outer canthus tilt line, per eye
        return [[P("left_eye_inner_corner"), P("left_eye_outer_corner")],
                [P("right_eye_inner_corner"), P("right_eye_outer_corner")]]
    if key == "fwhr":                      # width vs upper-face height
        return [[P("left_zygomatic"), P("right_zygomatic")],
                [P("eyebrows_bottom"), P("upper_lip_top_center")]]
    if key == "jaw_contour_deviation":     # jaw segment vs canthus-alare ref, both sides
        return [[P("left_jaw_angle1"), P("chin")],
                [P("left_eye_outer_corner"), P("left_alare_tip")],
                [P("right_jaw_angle1"), P("chin")],
                [P("right_eye_outer_corner"), P("right_alare_tip")]]
    if key == "jaw_contour_jaw_slope":     # jaw segment (gonion -> chin), per side
        return [[P("left_jaw_angle1"), P("chin")],
                [P("right_jaw_angle1"), P("chin")]]
    if key == "jaw_contour_canthus_alare_slope":   # canthus -> alare reference line
        return [[P("left_eye_outer_corner"), P("left_alare_tip")],
                [P("right_eye_outer_corner"), P("right_alare_tip")]]
    if key in ("upper_third", "middle_third", "lower_third"):   # the third's height
        top = {"upper_third": "top_center_forehead", "middle_third": "glabella",
               "lower_third": "base_of_nose"}[key]
        bot = {"upper_third": "glabella", "middle_third": "base_of_nose",
               "lower_third": "chin"}[key]
        return [[P(top), P(bot)]]
    if key == "bizygomatic_bigonial_ratio":
        return [[P("left_zygomatic"), P("right_zygomatic")],
                [P("left_jaw_angle2"), P("right_jaw_angle2")]]
    if key == "facial_fifths":
        return [[P("left_zygomatic"), P("right_zygomatic")],
                [P("left_eye_inner_corner"), P("left_eye_outer_corner")],
                [P("right_eye_inner_corner"), P("right_eye_outer_corner")]]
    if key == "inter_eye_ratio":
        return [[P("left_eye_inner_corner"), P("right_eye_inner_corner")],
                [P("left_eye_inner_corner"), P("left_eye_outer_corner")],
                [P("right_eye_inner_corner"), P("right_eye_outer_corner")]]
    if key == "orbitonasal_ratio":
        return [[P("left_alare_tip"), P("right_alare_tip")],
                [P("left_eye_inner_corner"), P("right_eye_inner_corner")]]
    if key == "nasofacial_proportion":
        return [[P("left_alare_tip"), P("right_alare_tip")],
                [P("left_zygomatic"), P("right_zygomatic")]]
    if key == "naso_oral_ratio":
        return [[P("lip_left_outer"), P("lip_right_outer")],
                [P("left_alare_tip"), P("right_alare_tip")]]
    if key == "face_golden_ratio":
        return [[P("top_center_forehead"), P("chin")],
                [P("left_zygomatic"), P("right_zygomatic")]]
    if key == "face_height_bigonial_width":
        return [[P("top_center_forehead"), P("chin")],
                [P("left_jaw_angle1"), P("right_jaw_angle1")]]
    if key == "lip_vermilion_ratio":
        return [[P("upper_lip_top_center"), P("upper_lip_bottom_center")],
                [P("lower_lip_top_center"), P("lower_lip_bottom_center")]]
    if key in ("lower_third_upper_split", "lower_third_lower_split"):
        return [[P("base_of_nose"), st], [st, P("chin")]]
    if key == "stomion_canthus_ratio":
        return [[st, P("chin")], [st, P("left_eye_outer_corner")],
                [st, P("right_eye_outer_corner")]]
    if key in ("ear_avg", "ear_asymmetry"):
        return [[P("left_upper_eyelid_center"), P("left_lower_eyelid_center")],
                [P("left_eye_inner_corner"), P("left_eye_outer_corner")],
                [P("right_upper_eyelid_center"), P("right_lower_eyelid_center")],
                [P("right_eye_inner_corner"), P("right_eye_outer_corner")]]
    return []


# Human labels for the Procrustes shape axes, PER SEX (the two PCAs differ, so
# male PC09 != female PC09). Derived from the deformation renders + the PC-vs-
# ratio correlation table (procrustes_features.correlate_axes_with_ratios).
# Cleanly-interpreted axes get real names; blended/weak ones are labelled
# honestly by their dominant correlate or "holistic shape" — not over-claimed.
# Shape axes are PLS components (supervised), ordered by relevance to the rating,
# so the low-numbered axes are the ones the model actually leans on. Names below
# come from each axis's deformation plot + its top-correlated ratios; the muddy
# tail (weak correlates) is left to the "Minor shape axis" fallback in _label.
SHAPE_LABELS = {
    "female": {
        "averageness": "Shape typicality",
        "shape_pls_01": "Lower-face length",
        "shape_pls_02": "Nose–mouth spacing & lip fullness",
        "shape_pls_03": "Eye spacing & face width",
        "shape_pls_04": "Jaw slope & asymmetry",
        "shape_pls_05": "Chin height & width-to-height",
        "shape_pls_06": "Lower-third balance",
        "shape_pls_09": "Face width & nose proportion",
    },
    "male": {
        "averageness": "Shape typicality",
        "shape_pls_01": "Jaw angularity & face width-to-height",
        "shape_pls_02": "Chin–mouth (lower-third) balance",
        "shape_pls_03": "Nose size & midface proportion",
        "shape_pls_05": "Cheekbone-to-jaw width",
        "shape_pls_08": "Face width & jaw contour",
        "shape_pls_11": "Facial asymmetry",
        "shape_pls_16": "Eye spacing & fifths",
    },
}


def _label(k, sex):
    """UI label for a feature key, using the per-sex shape names for shape axes."""
    if k in SHAPE_LABELS.get(sex, {}):
        return SHAPE_LABELS[sex][k]
    if k.startswith("shape_pls_"):        # unnamed (weakly-interpretable) PLS axis
        return "Minor shape axis"
    return LABELS.get(k, k)


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

    idx = 0 if sex == "male" else 1
    channel_keys = {"eye_r", "eye_g", "eye_b", "skin_r", "skin_g", "skin_b",
                    "lips_r", "lips_g", "lips_b"}
    items = [{
        "key": k,
        "label": _label(k, sex),
        # Only show a value where an ideal exists to read it against; shape axes
        # and colours are abstract, so value stays blank like the ideal.
        "value": _num(feats.get(k)) if k in IDEALS else None,
        "ideal": IDEALS[k][idx] if k in IDEALS else None,
        "contribution": round(float(c), 3),
        "landmarks": RATIO_LANDMARKS.get(k, []),
        "bar": _bar(k, feats.get(k), sex),
        "lines": ratio_lines(k, face_data),
    } for k, c in contribs.items() if k not in channel_keys]

    # Colour channels are meaningless individually -> sum each colour's 3 SHAP
    # contributions into one grouped "X colour" impact (SHAP is additive).
    for prefix, label in (("eye", "Eye colour"), ("skin", "Skin colour"),
                          ("lips", "Lip colour")):
        total = sum(float(contribs.get(f"{prefix}_{ch}", 0.0)) for ch in "rgb")
        items.append({"key": f"{prefix}_color", "label": label, "value": None,
                      "ideal": None, "contribution": round(total, 3),
                      "landmarks": [], "bar": None, "lines": []})

    contrib_items = sorted(items, key=lambda d: abs(d["contribution"]),
                           reverse=True)

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
