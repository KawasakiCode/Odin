"""
Final behaviour test: run Odin + the 3 SCUT benchmark CNNs across labelled face
folders (attractive / average / unattractive, per sex) and compare.

Usage (point it at one root that contains the sex/category subfolders, or list
several folders explicitly):
    python benchmark_faces.py  path/to/test_faces
    python benchmark_faces.py  folderA folderB ...

Sex and category are inferred from each file's path (keywords: male/female,
attractive/average/unattractive). Odin uses the RAW prediction (no male boost)
so it's directly comparable to the CNNs. Writes benchmark_results.csv and prints
per-group means + a correlation of image size vs each model's score.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_models"))

from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.trichion import apply_trichion
from Odin.Face_analysis.Ratios.appearance import appearance_features
from Odin.main import build_features, add_shape_features, MODEL_PATHS
import cnn_scorer

MODELS = {sex: joblib.load(p) for sex, p in MODEL_PATHS.items()}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".avif"}


def infer_sex(s):
    s = s.lower()
    if "female" in s or "women" in s or "woman" in s:   # check female before male
        return "female"
    if "male" in s or "men" in s or "man" in s:
        return "male"
    return None


def infer_cat(s):
    s = s.lower()
    if "unattractive" in s:              # check unattractive first
        return "unattractive"
    if "attractive" in s:
        return "attractive"
    if "average" in s or "avg" in s:
        return "average"
    return None


def odin_raw_score(img, px, sex):
    fd = extract_face_data(px)
    apply_trichion(fd, img)
    feats = build_features(fd, appearance_features(img, px))
    m = MODELS[sex]
    add_shape_features(feats, px, m)
    X = pd.DataFrame([feats], columns=m["feature_names"])
    return float(m["xgboost"].predict(X)[0])   # raw XGB output, no male boost


def analyze(path, sex):
    lm, img = calculate_landmarks_array(str(path))
    h, w = img.shape[:2]
    px = lm.copy()
    px[:, 0] *= w
    px[:, 1] *= h
    px[:, 2] *= w
    odin = odin_raw_score(img, px, sex)
    cnn = cnn_scorer.score_all(img, px[:, :2])
    return w, h, odin, cnn


def _run(roots):
    paths = sorted(p for root in roots for p in Path(root).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXT)
    print(f"found {len(paths)} images across {len(roots)} root(s)")

    rows = []
    for p in paths:
        sex = infer_sex(str(p))
        cat = infer_cat(str(p))
        if sex is None:
            # e.g. the "attractive" folder = attractive MALE (counterpart to
            # attractive_female); default a sexless folder to male.
            sex = "male"
        try:
            w, h, odin, cnn = analyze(p, sex)
        except Exception as e:
            print(f"  FAIL {p.name}: {e}")
            continue

        scores = {"odin": round(odin, 2), "alexnet": cnn["alexnet"],
                  "resnet18": cnn["resnet18"], "resnext50": cnn["resnext50"]}
        valid = {k: v for k, v in scores.items() if v is not None}
        best = max(valid, key=valid.get)

        def d(k):
            return round(scores[k] - odin, 2) if scores[k] is not None else None

        rows.append({
            "file": p.name, "sex": sex, "category": cat or "unknown",
            "width": w, "height": h, "megapixels": round(w * h / 1e6, 3),
            "odin": scores["odin"], "alexnet": scores["alexnet"],
            "resnet18": scores["resnet18"], "resnext50": scores["resnext50"],
            "d_alexnet": d("alexnet"), "d_resnet18": d("resnet18"),
            "d_resnext50": d("resnext50"),
            "best_model": best, "best_score": valid[best],
        })
        print(f"  {p.name:26} {sex:6} {cat or '?':12} {w}x{h:<5} "
              f"odin={scores['odin']:<5} ax={scores['alexnet']} "
              f"rn={scores['resnet18']} rx={scores['resnext50']}")

    if not rows:
        print("no rows produced")
        return

    df = pd.DataFrame(rows)
    out = ROOT / "benchmark_results.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {len(df)} rows -> {out}")

    cols = ["odin", "alexnet", "resnet18", "resnext50",
            "d_alexnet", "d_resnet18", "d_resnext50", "megapixels"]
    print("\n=== mean by sex + category ===")
    print(df.groupby(["sex", "category"])[cols].mean().round(2).to_string())

    print("\n=== correlation of image size (megapixels) with each score ===")
    for c in ["odin", "alexnet", "resnet18", "resnext50"]:
        sub = df.dropna(subset=[c])
        r = np.corrcoef(sub["megapixels"], sub[c])[0, 1] if len(sub) > 2 else float("nan")
        print(f"  {c:10} r = {r:+.3f}")


def main(roots):
    """Run the benchmark, sending all report output to benchmark_output.txt."""
    txt = ROOT / "benchmark_output.txt"
    with open(txt, "w", encoding="utf-8") as f, redirect_stdout(f):
        _run(roots)
    print(f"done -> {txt}  (+ benchmark_results.csv)")


if __name__ == "__main__":
    roots = sys.argv[1:] or [str(ROOT / "test_faces")]
    main(roots)
