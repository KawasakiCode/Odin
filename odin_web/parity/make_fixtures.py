"""
Build parity fixtures: run the reference Python pipeline on a sample of SCUT
faces and dump everything the TypeScript port must reproduce.

For each face it writes the decoded pixels (raw RGBA, so the JS side sees the
exact same bytes OpenCV did) and records the MediaPipe landmarks, the detected
hairline, region mask sizes, every model feature, the raw score and the
per-feature contributions. A few faces are also resized to larger,
non-square images to exercise the code paths SCUT's 350x350 squares don't.

Run from the repo root with the project venv (needs odin_training/scut/):
    venv/Scripts/python.exe odin_web/parity/make_fixtures.py [N_PER_GROUP]
"""
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Odin.Face_analysis.landmarks import calculate_landmarks_array  # noqa: E402
from Odin.Face_analysis.face_data import extract_face_data  # noqa: E402
from Odin.Face_analysis.trichion import apply_trichion  # noqa: E402
from Odin.Face_analysis.Ratios.appearance import appearance_features, _region_mask  # noqa: E402
from Odin.Face_analysis.Ratios.regions import extract_regions  # noqa: E402
from Odin.main import add_shape_features, build_features, MODEL_PATHS  # noqa: E402

SCUT = ROOT / "odin_training" / "scut" / "Images"
OUT = Path(__file__).resolve().parent / "fixtures"
MODELS = {sex: joblib.load(p) for sex, p in MODEL_PATHS.items()}


def clean(v):
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else v


def run(path, fid, sex):
    landmarks, img = calculate_landmarks_array(str(path))
    h, w = img.shape[:2]
    px = landmarks.copy()
    px[:, 0] *= w
    px[:, 1] *= h
    px[:, 2] *= w

    face_data = extract_face_data(px)
    trichion, reason = apply_trichion(face_data, img)
    regions = extract_regions(px)
    feats = build_features(face_data, appearance_features(img, px))
    model = MODELS[sex]
    add_shape_features(feats, px, model)

    X = pd.DataFrame([feats], columns=model["feature_names"])
    raw = float(model["xgboost"].predict(X)[0])
    dm = xgb.DMatrix(X, feature_names=model["feature_names"])
    contribs = model["xgboost"].get_booster().predict(dm, pred_contribs=True)[0]

    masks = {
        "lips": [regions["lips"]],
        "irises": [regions["left_iris"], regions["right_iris"]],
        "cheeks": [regions["left_cheek"], regions["right_cheek"]],
        "left_cheek": [regions["left_cheek"]],
    }
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    (OUT / f"{fid}.rgba").write_bytes(rgba.tobytes())

    return {
        "id": fid,
        "sex": sex,
        "width": w,
        "height": h,
        "landmarks": landmarks.tolist(),          # normalised, as MediaPipe returned them
        "trichion": None if trichion is None else [float(trichion[0]), float(trichion[1])],
        "trichionReason": reason,
        "forehead": [float(v) for v in face_data["top_center_forehead"][:2]],
        "regions": {k: v.tolist() for k, v in regions.items()},
        "maskCounts": {k: int((_region_mask(img.shape, polys) == 255).sum())
                       for k, polys in masks.items()},
        "features": {k: clean(feats.get(k)) for k in model["feature_names"]},
        "score": raw,
        "contribs": [float(c) for c in contribs[:-1]],
        "base": float(contribs[-1]),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    picks = []
    for group in ("AF", "AM", "CF", "CM"):
        files = sorted(SCUT.glob(f"{group}*.jpg"))
        picks += [files[i] for i in rng.choice(len(files), n, replace=False)]

    fixtures = []
    for p in picks:
        sex = "female" if p.name[1] == "F" else "male"
        try:
            fixtures.append(run(p, p.stem, sex))
        except ValueError as e:                 # no face found
            print("skip", p.name, e)

    # Larger, non-square variants (upscaled + aspect change) of a few faces.
    tmp = OUT / "_resized.png"
    for p in picks[::n][:4]:
        img = cv2.imread(str(p))
        big = cv2.resize(img, (760, 620), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(tmp), big)
        sex = "female" if p.name[1] == "F" else "male"
        fixtures.append(run(tmp, f"{p.stem}_760x620", sex))
    tmp.unlink()

    (OUT / "fixtures.json").write_text(json.dumps(fixtures))
    print(f"wrote {len(fixtures)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
