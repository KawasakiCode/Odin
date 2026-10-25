"""
Regression tests for the inference pipeline on the reference faces.

1. Feature code (deterministic): from the recorded landmarks and pixels, every
   feature, the score and the contributions must reproduce the recorded values.
   Any change to a ratio, the hairline walk, the colour features or the shape
   model shows up here.
2. End to end (needs the MediaPipe model): re-detecting landmarks must land on
   the recorded ones and give the same score. MediaPipe isn't bit-identical
   across CPUs/OSes, so this allows a little drift.
"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from conftest import needs_landmarker
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.Ratios.appearance import appearance_features
from Odin.Face_analysis.trichion import apply_trichion
from Odin.main import add_shape_features, build_features


def run_pipeline(img, px, model):
    fd = extract_face_data(px)
    appearance = appearance_features(img, px)
    apply_trichion(fd, img)
    feats = build_features(fd, appearance)
    add_shape_features(feats, px, model)
    X = pd.DataFrame([feats], columns=model["feature_names"])
    score = float(model["xgboost"].predict(X)[0])
    dm = xgb.DMatrix(X, feature_names=model["feature_names"])
    contribs = model["xgboost"].get_booster().predict(dm, pred_contribs=True)[0]
    return feats, score, contribs, fd


def test_features_match_recorded(face, models):
    rec = face["record"]
    model = models[rec["sex"]]
    feats, score, contribs, fd = run_pipeline(face["img"], face["px"], model)

    for k, expected in rec["features"].items():
        got = feats.get(k)
        if expected is None:
            assert got is None or np.isnan(got), k
        else:
            assert got == pytest.approx(expected, rel=1e-9, abs=1e-9), k
    assert fd["top_center_forehead"][:2] == pytest.approx(rec["forehead"], abs=1e-9)
    assert score == pytest.approx(rec["score"], abs=1e-5)
    assert contribs[:-1] == pytest.approx(rec["contribs"], abs=1e-5)
    # contributions are exact: they sum to the prediction
    assert contribs.sum() == pytest.approx(score, abs=1e-4)


@needs_landmarker
def test_end_to_end_matches_recorded(face, models):
    rec = face["record"]
    lm, img = calculate_landmarks_array(str(face["path"]))
    h, w = img.shape[:2]
    drift = np.hypot((lm[:, 0] - np.array(rec["landmarks"])[:, 0]) * w,
                     (lm[:, 1] - np.array(rec["landmarks"])[:, 1]) * h)
    assert drift.mean() < 0.05, f"mean landmark drift {drift.mean():.3f}px"
    px = lm.copy()
    px[:, 0] *= w
    px[:, 1] *= h
    px[:, 2] *= w
    _, score, _, _ = run_pipeline(img, px, models[rec["sex"]])
    assert score == pytest.approx(rec["score"], abs=0.1)
