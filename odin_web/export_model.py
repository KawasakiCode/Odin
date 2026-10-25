"""
Export the trained Odin bundles to plain JSON for the in-browser build.

The .joblib bundles hold Python objects (XGBRegressor, PLSRegression) that a
browser can't load, so this flattens everything inference needs into arrays:

  - the XGBoost trees (children, split feature/threshold, default direction,
    leaf values, and node covers for TreeSHAP) + base score,
  - the Procrustes reference shape and only the PLS rotation columns for the
    shape axes the model keeps (the unnamed ones are dropped at reindex anyway),
  - the averageness quantiles the UI ranks a face against.

Also copies the MediaPipe face_landmarker.task next to them.

Run from the repo root with the project venv:
    venv/Scripts/python.exe odin_web/export_model.py [--out DIR]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "public" / "models"


def export_trees(xgb_model):
    raw = json.loads(xgb_model.get_booster().save_raw("json"))
    learner = raw["learner"]
    assert learner["gradient_booster"]["name"] == "gbtree"
    assert learner["objective"]["name"] == "reg:squarederror"
    # XGBoost 3.x stores base_score as a vector string, e.g. "[5.2852E0]".
    base_score = float(learner["learner_model_param"]["base_score"].strip("[]"))

    trees = []
    for t in learner["gradient_booster"]["model"]["trees"]:
        # Values come straight from XGBoost's JSON (shortest float32 repr), so
        # Math.fround() in JS recovers the exact float32 thresholds.
        trees.append({
            "left": t["left_children"],
            "right": t["right_children"],
            "feature": t["split_indices"],
            "value": t["split_conditions"],     # threshold, or leaf value on leaves
            "defaultLeft": [int(v) for v in t["default_left"]],
            "cover": t["sum_hessian"],
        })
    return base_score, trees


def export_shape(bundle):
    pls = bundle["shape_pls"]
    keep = [f for f in bundle["feature_names"] if f.startswith("shape_pls_")]
    rot = np.asarray(pls.x_rotations_, dtype=np.float64)
    return {
        "refMean": np.asarray(bundle["shape_ref_mean"], dtype=np.float64).ravel().tolist(),
        "xMean": np.asarray(pls._x_mean, dtype=np.float64).tolist(),
        "xStd": np.asarray(pls._x_std, dtype=np.float64).tolist(),
        # feature name -> rotation column (shape_pls_01 is column 0, ...)
        "components": {f: rot[:, int(f.split("_")[-1]) - 1].tolist() for f in keep},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (default: public/models)")
    out_dir = ap.parse_args(argv).out
    out_dir.mkdir(parents=True, exist_ok=True)
    for sex in ("male", "female"):
        bundle = joblib.load(ROOT / "models" / f"model_{sex}.joblib")
        xgb_model = bundle["xgboost"]
        if getattr(xgb_model, "best_iteration", None) is not None:
            sys.exit("model was trained with early stopping; export would need "
                     "to truncate to best_iteration")
        base_score, trees = export_trees(xgb_model)
        out = {
            "label": bundle["label"],
            "featureNames": list(bundle["feature_names"]),
            "baseScore": base_score,
            "trees": trees,
            "xgbPredMean": bundle.get("xgb_pred_mean"),
            "xgbPredMax": bundle.get("xgb_pred_max"),
            "averagenessQuantiles": [float(v) for v in bundle["averageness_quantiles"]],
            "shape": export_shape(bundle),
        }
        path = out_dir / f"model_{sex}.json"
        path.write_text(json.dumps(out, separators=(",", ":")))
        print(f"{path}: {len(trees)} trees, "
              f"{path.stat().st_size / 1e6:.1f} MB")

    task = ROOT / "models" / "face_landmarker.task"
    if not task.exists():
        sys.exit(f"missing {task} — download it first (see the main README)")
    shutil.copy(task, out_dir / "face_landmarker.task")
    print(f"copied {task.name}")


if __name__ == "__main__":
    main()
