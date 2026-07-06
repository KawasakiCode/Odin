"""
SCUT-FBP5500 data loader.

  - Images live in a single flat folder (scut/Images/*.jpg).
  - Labels come from scut/scut_ratings.csv (Image_ID + Attractiveness), already
    averaged over the 60 raters and rescaled from the 1-5 SCUT scale to 1-10.
    Build that CSV once with build_scut_ratings() if it is missing.
  - Filenames encode race+sex: AF/AM = Asian, CF/CM = Caucasian, second letter
    is sex. e.g. "AF123.jpg" -> race group "AF".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tqdm import tqdm
import cv2

from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.Ratios.appearance import appearance_features
from Odin.Face_analysis.Ratios.calculate_ratios import (
    bizygomatic_bigonial_ratio, canthal_tilt_final, eye_aspect_ratio,
    face_golden_ratio, face_height_bigonial_width, facial_fifths,
    frontal_jaw_contour_angle, fwhr, height_ratio_36, horizontal_thirds,
    lip_vermilion_ratio, lower_third_split, naso_oral_ratio,
    nasofacial_proportion, orbitonasal_ratio, stomion_canthus_ratio,
    symmetry_score, width_ratio_46,
)
# Reuse the overlay helper so SCUT extraction also dumps debug overlays.
from Odin.Face_analysis.trichion import apply_trichion
from overlay import save_landmark_overlay, DEBUG_OVERLAY_DIR

# Anchor data paths to this file's folder so the script works from any working
# directory (repo root, IDE run button, etc.), not just when run inside odin_model.
BASE = Path(__file__).resolve().parent
RATINGS_XLSX = str(BASE / "scut" / "All_Ratings.xlsx")
RATINGS_CSV  = str(BASE / "scut" / "scut_ratings.csv")
IMAGES_DIR   = str(BASE / "scut" / "Images")
CACHE_CSV    = str(BASE / "training_data_scut.csv")
TASK_MODEL   = str(BASE.parent / "models" / "face_landmarker.task")


def build_scut_ratings(xlsx_path=RATINGS_XLSX, out_csv=RATINGS_CSV):
    """
    Collapse the per-rater SCUT ratings into one average score per image and
    rescale 1-5 -> 1-10, writing Image_ID + Attractiveness to out_csv.

    Linear rescale: scaled = (raw - 1) / 4 * 9 + 1, so 1->1 and 5->10.
    """
    x = pd.read_excel(xlsx_path)
    g = x.groupby("Filename")["Rating"].mean().reset_index()
    g.columns = ["Image_ID", "Attractiveness"]
    g["Attractiveness"] = (g["Attractiveness"] - 1) / 4 * 9 + 1
    g.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}: {len(g)} faces, "
          f"scaled range {g['Attractiveness'].min():.2f}-{g['Attractiveness'].max():.2f}")
    return g


def process_scut_images(images_dir=IMAGES_DIR):
    """Run MediaPipe + the shared feature extraction over the flat SCUT folder."""
    base_dir = Path(images_dir)
    all_image_paths = sorted(base_dir.glob("*.jpg"))
    print(f"Found {len(all_image_paths)} SCUT images. Booting up MediaPipe...")

    base_options = python.BaseOptions(model_asset_path=TASK_MODEL)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, output_face_blendshapes=False, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    extracted_data = []
    DEBUG_OVERLAY_DIR.mkdir(exist_ok=True)

    for idx, img_path in enumerate(tqdm(all_image_paths, desc="Extracting (SCUT)")):
        # SCUT id IS the filename, e.g. "AF123.jpg" — matches the ratings CSV.
        image_id = img_path.name

        try:
            mp_image = mp.Image.create_from_file(str(img_path))
        except Exception as e:
            print(f"Failed to load image {img_path.name}: {e}")
            continue

        detection_result = detector.detect(mp_image)
        if len(detection_result.face_landmarks) == 0:
            print(f"No face detected in {img_path.name}")
            continue

        try:
            face_landmarks = detection_result.face_landmarks[0]
            iw, ih = mp_image.width, mp_image.height
            landmarks_array = np.array(
                [[lm.x * iw, lm.y * ih, lm.z * iw] for lm in face_landmarks])
            face_data = extract_face_data(landmarks_array)

            img_bgr = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)

            trichion, reason = apply_trichion(face_data, img_bgr)

            if idx % 100 == 0:
                try:
                    save_landmark_overlay(
                        mp_image, landmarks_array, face_data,
                        DEBUG_OVERLAY_DIR / f"scut_{image_id}_overlay.jpg")
                    print(f"Overlay: {image_id}, reason={reason or 'detected'}")
                except Exception as e:
                    print(f"Overlay failed for {img_path.name}: {e}")

            jc  = frontal_jaw_contour_angle(face_data)
            ff  = facial_fifths(face_data)
            ht  = horizontal_thirds(face_data)
            ear = eye_aspect_ratio(face_data)
            ls  = lower_third_split(face_data)
            app = appearance_features(img_bgr, landmarks_array)

            ratios = {
                "Image_ID": image_id,
                "symmetry": symmetry_score(face_data),
                "width_ratio_46": width_ratio_46(face_data),
                "height_ratio_36": height_ratio_36(face_data),
                "canthal_average": canthal_tilt_final(face_data),
                "fwhr": fwhr(face_data),
                "jaw_contour_deviation": jc["deviation"],
                "jaw_contour_jaw_slope": jc["jaw_slope"],
                "jaw_contour_canthus_alare_slope": jc["canthus_alare_slope"],
                "upper_third": ht["upper_perc"],
                "middle_third": ht["middle_perc"],
                "lower_third": ht["lower_perc"],
                "bizygomatic_bigonial_ratio": bizygomatic_bigonial_ratio(face_data),
                "facial_fifths": ff["fifths_ratio"],
                "inter_eye_ratio": ff["inter_eye_ratio"],
                "orbitonasal_ratio": orbitonasal_ratio(face_data),
                "nasofacial_proportion": nasofacial_proportion(face_data),
                "naso_oral_ratio": naso_oral_ratio(face_data),
                "face_golden_ratio": face_golden_ratio(face_data),
                "face_height_bigonial_width": face_height_bigonial_width(face_data),
                "lip_vermilion_ratio": lip_vermilion_ratio(face_data),
                "lower_third_upper_split": ls["upper_pct"],
                "lower_third_lower_split": ls["lower_pct"],
                "stomion_canthus_ratio": stomion_canthus_ratio(face_data),
                "ear_avg": ear["average"],
                "ear_asymmetry": ear["asymmetry"],
                "skin_texture": app["skin_texture"],
                "lips_r": app["lips_r"], "lips_g": app["lips_g"], "lips_b": app["lips_b"],
                "eye_r": app["eye_r"], "eye_g": app["eye_g"], "eye_b": app["eye_b"],
                "skin_r": app["skin_r"], "skin_g": app["skin_g"], "skin_b": app["skin_b"],
                "eye_skin_luminance_contrast": app["eye_skin_luminance_contrast"],
                "lip_skin_luminance_contrast": app["lip_skin_luminance_contrast"],
                "lip_skin_redness_contrast": app["lip_skin_redness_contrast"],
                "eye_skin_redness_contrast": app["eye_skin_redness_contrast"],
                "facial_contrast_avg": app["facial_contrast_avg"],
            }
            extracted_data.append(ratios)
        except Exception as e:
            print(f"Failed ratio extraction on {img_path.name}: {e}")
            continue

    detector.close()
    ratios_df = pd.DataFrame(extracted_data)
    print(f"\n--- SCUT extraction complete: {len(ratios_df)} faces ---")
    return ratios_df


def split_by_gender(cache_path=CACHE_CSV):
    """
    Build (or load) the merged SCUT dataset and split into female/male sets,
    mirroring data.split_by_gender so train.py can swap loaders seamlessly.

    Returns (X_female, X_male, y_female, y_male) on the raw 1-10 Attractiveness.
    """
    if Path(cache_path).exists():
        print("SCUT cache found, loading directly...")
        df = pd.read_csv(cache_path)
        print(f"Loaded {len(df)} faces, {df.shape[1]} columns from cache")
    else:
        if not Path(RATINGS_CSV).exists():
            build_scut_ratings()
        ratings = pd.read_csv(RATINGS_CSV)
        ratios_df = process_scut_images()
        df = pd.merge(ratios_df, ratings, on="Image_ID", how="inner")
        print(f"Merged: {len(df)} faces "
              f"({len(ratios_df)} extracted, {len(ratios_df) - len(df)} without a label)")
        df.to_csv(cache_path, index=False)

    # race+sex group from the filename prefix (AF123.jpg -> "AF"), and sex.
    df["race"] = df["Image_ID"].str.extract(r"^([A-Z]{2})")
    df["Gender"] = df["race"].str[1].map({"F": 0, "M": 1})

    # Race-controlled target (kept for analysis / optional confound-free runs).
    df["Attractiveness_resid"] = (
        df["Attractiveness"] - df.groupby("race")["Attractiveness"].transform("mean"))

    df_female = df[df["Gender"] == 0].copy()
    df_male   = df[df["Gender"] == 1].copy()
    print(f"Split by gender: {len(df_female)} female, {len(df_male)} male")

    # Drop non-feature columns; train on the raw 1-10 score for readable output.
    # X/y are indexed by Image_ID so train.py can join the landmark cache for the
    # Procrustes shape features.
    drop_cols = ["Image_ID", "Attractiveness", "Attractiveness_resid",
                 "Gender", "race"]

    def _xy(d):
        ids = pd.Index(d["Image_ID"].values, name="Image_ID")
        X = d.drop(columns=drop_cols, errors="ignore")
        X.index = ids
        y = pd.Series(d["Attractiveness"].values, index=ids)
        return X, y

    X_female, y_female = _xy(df_female)
    X_male,   y_male   = _xy(df_male)
    return X_female, X_male, y_female, y_male


if __name__ == "__main__":
    build_scut_ratings()
