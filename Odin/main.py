"""
Odin entry point.

Reads the photo at constants.IMAGEPATH, extracts MediaPipe landmarks, computes
the geometric ratios (calculate_ratios) and the colour/texture appearance
features (appearance), assembles them into the exact 40-feature vector the
trained model expects, and prints the XGBoost attractiveness score (1-10). The
model is trained directly on the 1-10 scale (SCUT labels span ~1.04-9.44, mean
~5.5), so the raw prediction is reported as-is — no rescaling. (RandomForest was
dropped: as a leaf-averaging model it regresses the tails toward the mean and
can't follow the attractiveness curve; XGBoost tracks it.)

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

    # Training (data_scut.py) computes EVERY feature on pixel-space landmarks
    # (lm.x*w, lm.y*h, lm.z*w), so we must too — otherwise the angle/ratio
    # features differ on non-square photos (they coincide only on square images
    # like SCUT's 350x350). Scale the normalised mesh to this image once and use
    # it for both the geometric ratios and the appearance sampling.
    h, w = image_bgr.shape[:2]
    pixel_landmarks = landmarks.copy()
    pixel_landmarks[:, 0] *= w
    pixel_landmarks[:, 1] *= h
    pixel_landmarks[:, 2] *= w

    face_data = extract_face_data(pixel_landmarks)
    appearance = appearance_features(image_bgr, pixel_landmarks)

    features = build_features(face_data, appearance)

    model = joblib.load(MODEL_PATHS[SEX])
    feature_names = model["feature_names"]
    # Order the columns to exactly match what the models were trained on.
    X = pd.DataFrame([features], columns=feature_names)

    score = float(model["xgboost"].predict(X)[0])

    print(f"Attractiveness ({model['label']}): {score:.2f} / 10")


if __name__ == "__main__":
    main()
