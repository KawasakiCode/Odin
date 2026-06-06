"""
Odin entry point.

Reads the photo at constants.IMAGEPATH, extracts MediaPipe landmarks, computes
the geometric ratios (calculate_ratios) and the colour/texture appearance
features (appearance), assembles them into the exact 40-feature vector the
trained models expect, and prints an attractiveness score on a 1-10 scale from
both the RandomForest and the XGBoost regressor.

Can be run either as a module from the project root (python -m Odin.main) or
directly (python Odin/main.py / the IDE run button): the sys.path bootstrap
below puts the project root on the import path so the absolute `Odin.*` imports
resolve in both cases.
"""
import sys
from pathlib import Path

# Project root = the directory that contains the `Odin` package (one level up
# from this file). Ensure it is importable before any `from Odin...` import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.constants import IMAGEPATH, SEX
from Odin.Face_analysis.Ratios.calculate_ratios import (
    bizygomatic_bigonial_ratio, canthal_tilt_final, eye_aspect_ratio,
    face_golden_ratio, face_height_bigonial_width, facial_fifths,
    frontal_jaw_contour_angle, fwhr, height_ratio_36, horizontal_thirds,
    lip_vermilion_ratio, lower_third_split, naso_oral_ratio,
    nasofacial_proportion, orbitonasal_ratio, stomion_canthus_ratio,
    symmetry_score, width_ratio_46,
)
from Odin.Face_analysis.Ratios.appearance import appearance_features

MODEL_PATHS = {
    "male": PROJECT_ROOT / "models" / "model_male.joblib",
    "female": PROJECT_ROOT / "models" / "model_female.joblib",
}


def to_ten(raw):
    """Map the model's 1-7 prediction onto a 1-10 scale (1->1, 7->10)."""
    return 1 + (raw - 1) * 1.5


def build_features(face_data, appearance):
    """
    Flatten the ratio + appearance results into the {name: value} feature map
    the models were trained on. The functions that return a dict are unpacked
    here into their individual named features.
    """
    jaw = frontal_jaw_contour_angle(face_data)
    thirds = horizontal_thirds(face_data)
    fifths = facial_fifths(face_data)
    lower = lower_third_split(face_data)
    ear = eye_aspect_ratio(face_data)

    features = {
        "symmetry": symmetry_score(face_data),
        "width_ratio_46": width_ratio_46(face_data),
        "height_ratio_36": height_ratio_36(face_data),
        "canthal_average": canthal_tilt_final(face_data),
        "fwhr": fwhr(face_data),
        "jaw_contour_deviation": jaw["deviation"],
        "jaw_contour_jaw_slope": jaw["jaw_slope"],
        "jaw_contour_canthus_alare_slope": jaw["canthus_alare_slope"],
        "upper_third": thirds["upper_perc"],
        "middle_third": thirds["middle_perc"],
        "lower_third": thirds["lower_perc"],
        "bizygomatic_bigonial_ratio": bizygomatic_bigonial_ratio(face_data),
        "facial_fifths": fifths["fifths_ratio"],
        "inter_eye_ratio": fifths["inter_eye_ratio"],
        "orbitonasal_ratio": orbitonasal_ratio(face_data),
        "nasofacial_proportion": nasofacial_proportion(face_data),
        "naso_oral_ratio": naso_oral_ratio(face_data),
        "face_golden_ratio": face_golden_ratio(face_data),
        "face_height_bigonial_width": face_height_bigonial_width(face_data),
        "lip_vermilion_ratio": lip_vermilion_ratio(face_data),
        "lower_third_upper_split": lower["upper_pct"],
        "lower_third_lower_split": lower["lower_pct"],
        "stomion_canthus_ratio": stomion_canthus_ratio(face_data),
        "ear_avg": ear["average"],
        "ear_asymmetry": ear["asymmetry"],
    }
    # appearance_features already keys its output to the remaining feature
    # names (skin_texture, lips_r/g/b, eye_*, skin_*, *_contrast, ...).
    features.update(appearance)
    return features


def main():
    landmarks, image_bgr = calculate_landmarks_array(IMAGEPATH)

    face_data = extract_face_data(landmarks)

    # Appearance samples raw pixels, so it needs pixel-space landmarks. The
    # mesh is normalised (x by width, y by height); scale it to this image.
    h, w = image_bgr.shape[:2]
    pixel_landmarks = landmarks.copy()
    pixel_landmarks[:, 0] *= w
    pixel_landmarks[:, 1] *= h
    pixel_landmarks[:, 2] *= w
    appearance = appearance_features(image_bgr, pixel_landmarks)

    features = build_features(face_data, appearance)

    model = joblib.load(MODEL_PATHS[SEX])
    feature_names = model["feature_names"]
    # Order the columns to exactly match what the models were trained on.
    X = pd.DataFrame([features], columns=feature_names)

    rf_score = to_ten(float(model["random_forest"].predict(X)[0]))
    xgb_score = to_ten(float(model["xgboost"].predict(X)[0]))

    print(f"Attractiveness score ({model['label']}), 1-10 scale:")
    print(f"  RandomForest : {rf_score:.2f}")
    print(f"  XGBoost      : {xgb_score:.2f}")


if __name__ == "__main__":
    main()
