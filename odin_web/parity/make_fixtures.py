"""
Build parity fixtures: run the reference Python pipeline on a set of faces and
dump everything the TypeScript port must reproduce.

For each face it writes the decoded pixels (lossless PNG, so the JS side sees
the exact bytes OpenCV decoded) and records the MediaPipe landmarks, the
detected hairline, region mask sizes, every model feature, the raw score and
the per-feature contributions.

Two sets:
  --set scut     a random sample of SCUT faces + 4 resized to non-square 760x620.
                 Local only: SCUT pixels must never be committed (parity/fixtures/).
  --set samples  the public-domain AI sample faces in public/samples/ plus resized,
                 rotated and mirrored variants. Safe to commit (parity/fixtures_ci/);
                 this is what CI runs.

Run from the repo root with the project venv:
    venv/Scripts/python.exe odin_web/parity/make_fixtures.py --set samples
    venv/Scripts/python.exe odin_web/parity/make_fixtures.py --set scut [--n 8]
"""
import argparse
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
WEB = Path(__file__).resolve().parents[1]
# public-domain sample faces and their sexes (mirrors src/samples.ts)
SAMPLES = {"man-1": "male", "man-2": "male", "woman-1": "female", "woman-2": "female"}
MODELS = {sex: joblib.load(p) for sex, p in MODEL_PATHS.items()}


def clean(v):
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else v


def run(path, fid, sex, out):
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
    cv2.imwrite(str(out / f"{fid}.png"), img)          # lossless: same pixels in JS

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


def scut_set(n, out):
    rng = np.random.default_rng(0)
    picks = []
    for group in ("AF", "AM", "CF", "CM"):
        files = sorted(SCUT.glob(f"{group}*.jpg"))
        picks += [files[i] for i in rng.choice(len(files), n, replace=False)]
    jobs = [(cv2.imread(str(p)), p.stem, "female" if p.name[1] == "F" else "male") for p in picks]
    # larger, non-square variants (upscaled + aspect change) of a few faces
    jobs += [(cv2.resize(img, (760, 620), interpolation=cv2.INTER_CUBIC), f"{fid}_760x620", sex)
             for img, fid, sex in jobs[::n][:4]]
    return jobs


def samples_set(out):
    jobs = []
    for name, sex in SAMPLES.items():
        img = cv2.imread(str(WEB / "public" / "samples" / f"{name}.jpg"))
        h, w = img.shape[:2]
        jobs.append((img, name, sex))
        jobs.append((cv2.flip(img, 1), f"{name}_mirrored", sex))
        if name == "woman-1":       # non-square, larger: kernel sizes, aspect
            jobs.append((cv2.resize(img, (900, 700), interpolation=cv2.INTER_CUBIC), f"{name}_900x700", sex))
        if name == "man-1":         # in-plane rotation: hairline walk, hull, roll
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)
            jobs.append((cv2.warpAffine(img, M, (w, h), borderValue=(40, 40, 40)), f"{name}_rot12", sex))
        if name == "woman-2":       # face cut off by the frame: off-image regions, clipping
            jobs.append((img[:, int(w * 0.33):].copy(), f"{name}_cropped", sex))
        if name == "man-2":         # small image: kernel larger than features, borders
            jobs.append((cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA), f"{name}_300", sex))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("scut", "samples"), default="samples")
    ap.add_argument("--n", type=int, default=8, help="SCUT faces per race/sex group")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / ("fixtures" if args.set == "scut" else "fixtures_ci")
    out.mkdir(parents=True, exist_ok=True)
    jobs = scut_set(args.n, out) if args.set == "scut" else samples_set(out)

    fixtures = []
    tmp = out / "_input.png"
    for img, fid, sex in jobs:
        cv2.imwrite(str(tmp), img)                  # the pipeline reads from a path
        try:
            fixtures.append(run(tmp, fid, sex, out))
        except ValueError as e:                     # no face found
            print("skip", fid, e)
    tmp.unlink()
    (out / "fixtures.json").write_text(json.dumps(fixtures))
    print(f"wrote {len(fixtures)} fixtures to {out}")


if __name__ == "__main__":
    main()
