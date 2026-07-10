import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
import xgboost as xgb

import joblib
from pathlib import Path

# Training on SCUT-FBP5500 (1-10 range).
from data_scut import split_by_gender
from shape_utils import (load_landmarks, fit_shape_model, shape_feature_matrix,
                         SHAPE_NAMES)

# Only XGBoost is trained. RandomForest was dropped: as a leaf-averaging model
# it regresses the tails toward the crowd mean (an 8/10 face gets dragged to
# ~6-7) and cannot follow the attractiveness curve. Gradient boosting chases
# the residuals and tracks the full 1-10 range.
#
# Features = the 40 ratios/appearance values + Procrustes shape features
# (averageness + 15 shape-PCA). The consensus mean shape and PCA basis are fit
# here and stored in each bundle so inference reproduces the features exactly.

RANDOM_STATE = 42
# Anchor outputs to this file's folder so train.py works from any CWD.
BASE = Path(__file__).resolve().parent

# Final models keep ONLY shape axes that have an interpretable name (mirrors
# odin_ui SHAPE_LABELS): the unnamed "Minor shape axis" PLS modes are dropped so
# every feature in the deployed model is explainable. This costs a small, measured
# amount of R² (the unnamed modes carry diffuse holistic signal) — a deliberate
# interpretability-over-accuracy choice. The PLS basis is still fit at 25
# components per fold; we simply don't feed the unnamed score columns to XGBoost.
NAMED_SHAPE = {
    "FEMALE": ["averageness"] + [f"shape_pls_{i:02d}" for i in (1, 2, 3, 4, 5, 6, 9)],
    "MALE":   ["averageness"] + [f"shape_pls_{i:02d}" for i in (1, 2, 3, 5, 8, 11, 16)],
}


def load_features():
    """
    Ratios + the raw landmarks per sex. Shape features are NOT baked in here:
    the shape model is PLS (supervised), so it must be fit on each fold's train
    split to avoid leaking the labels. Callers add shape features via augment()
    after they have chosen a fit split. Returns (X_ratios, y, raw) per sex.
    """
    Xf, Xm, yf, ym = split_by_gender()
    id2lm = load_landmarks()

    def prep(X, y):
        ids = [i for i in X.index if i in id2lm]
        X, y = X.loc[ids], y.loc[ids]
        raw = np.stack([id2lm[i] for i in ids]).astype(np.float64)
        return X, y, raw

    return prep(Xf, yf), prep(Xm, ym)


def augment(X_ratios, raw, M, proj, label):
    """Concatenate shape features onto the ratio matrix, keeping only the named
    (interpretable) shape axes for this sex — see NAMED_SHAPE. The shape model
    (M, proj) is fit elsewhere (on the train split)."""
    shp = pd.DataFrame(shape_feature_matrix(raw, M, proj),
                       columns=SHAPE_NAMES, index=X_ratios.index)
    return pd.concat([X_ratios, shp[NAMED_SHAPE[label]]], axis=1)


def report(name, fitted_model, X_test, y_test, feature_names):
    """Print held-out metrics and feature importances for a fitted model."""
    y_pred = fitted_model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print(f"\n--- {name} performance (held-out 20% test set) ---")
    print(f"R^2 : {r2:.3f}  (fraction of attractiveness variance explained)")
    print(f"MAE : {mae:.3f}  (avg absolute error, on the 1-10 score)")
    print(f"RMSE: {rmse:.3f}")

    print(f"\n--- {name} top feature importance ---")
    importances = pd.Series(
        fitted_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    for fname, imp in importances.head(12).items():
        print(f"{fname:35s} {imp:6.3f}")


def train_models(label, X_ratios, y, raw):
    """Train and evaluate XGBoost on one gender subset, then export it.

    The held-out metric is honest: the PLS shape model is fit on the TRAIN split
    only and applied to the test split, so the shape features never see the test
    labels. The exported bundle (see export_model) instead fits shape on all data.
    """
    n_feat = X_ratios.shape[1] + len(NAMED_SHAPE[label])
    print("\n" + "=" * 70)
    print(f"  {label}  ({len(X_ratios)} faces, {n_feat} features)")
    print("=" * 70)

    idx = np.arange(len(X_ratios))
    tr, te = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE)

    # Shape model fit on TRAIN only, then applied to both splits.
    M, proj = fit_shape_model(raw[tr], y.iloc[tr].values)
    X_train = augment(X_ratios.iloc[tr], raw[tr], M, proj, label)
    X_test = augment(X_ratios.iloc[te], raw[te], M, proj, label)
    y_train, y_test = y.iloc[tr], y.iloc[te]

    # Early stopping needs a validation set carved from TRAIN only (never the
    # test set, which would leak it and inflate the score).
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=RANDOM_STATE
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.7, reg_lambda=2, early_stopping_rounds=50,
        random_state=RANDOM_STATE,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"\n{label} — XGBoost stopped at {xgb_model.best_iteration + 1} trees "
          f"(of 1000 max; best val score {xgb_model.best_score:.3f})")
    report(f"{label} — XGBoost", xgb_model, X_test, y_test, X_train.columns)

    export_model(label, X_ratios, y, raw, xgb_model.best_iteration + 1)


def export_model(label, X_ratios, y, raw, n_estimators):
    """Refit the shape model + XGB on ALL rows and save to models/.

    Fitting shape on all training data here is correct — the deployed model uses
    every face it has; only the CV metric above needs per-fold shape fitting. The
    feature column order, target, male-boost calibration stats, and the shape
    model (consensus mean + PLS projection) are bundled so inference reproduces
    the exact feature vector.
    """
    MODEL_DIR = BASE.parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    slug = label.lower().replace(" ", "_")

    M, proj = fit_shape_model(raw, y.values)
    X = augment(X_ratios, raw, M, proj, label)

    xgb_full = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=2,
        random_state=RANDOM_STATE,
    )
    xgb_full.fit(X, y)

    # Calibration stats for the presentation-layer male boost (see main.py).
    preds = xgb_full.predict(X)

    bundle = {
        "xgboost": xgb_full,
        "feature_names": list(X.columns),
        "target": "Attractiveness",
        "label": label,
        "n_samples": len(X),
        "xgb_pred_mean": float(preds.mean()),
        "xgb_pred_max": float(preds.max()),
        "shape_ref_mean": M,
        "shape_pls": proj,
    }
    out = MODEL_DIR / f"model_{slug}.joblib"
    joblib.dump(bundle, out)
    print(f"\n{label} — saved trained model -> {out} "
          f"({len(X)} rows, {X.shape[1]} features)")


def cross_validate(label, X_ratios, y, raw, n_splits=5):
    """Honest K-fold CV: the PLS shape model is refit on each fold's train split,
    so nothing leaks. This is the number to trust / cite (report()'s single split
    is just a quick fit-check)."""
    print("\n" + "#" * 70)
    print(f"  {label}  —  {n_splits}-fold cross-validation  ({len(X_ratios)} faces)")
    print("#" * 70)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = {"r2": [], "mae": [], "rmse": []}
    for train_idx, test_idx in kf.split(X_ratios):
        M, proj = fit_shape_model(raw[train_idx], y.iloc[train_idx].values)
        X_train = augment(X_ratios.iloc[train_idx], raw[train_idx], M, proj, label)
        X_test = augment(X_ratios.iloc[test_idx], raw[test_idx], M, proj, label)
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.20, random_state=RANDOM_STATE)
        m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=2, early_stopping_rounds=50,
            random_state=RANDOM_STATE)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred = m.predict(X_test)
        scores["r2"].append(r2_score(y_test, pred))
        scores["mae"].append(mean_absolute_error(y_test, pred))
        scores["rmse"].append(mean_squared_error(y_test, pred) ** 0.5)

    r2, mae, rmse = (np.array(scores[k]) for k in ("r2", "mae", "rmse"))
    print(f"\n--- {label} — XGBoost ({n_splits}-fold CV) ---")
    print(f"R^2 : {r2.mean():.3f} ± {r2.std():.3f}   per-fold: {np.round(r2, 3)}")
    print(f"MAE : {mae.mean():.3f} ± {mae.std():.3f}")
    print(f"RMSE: {rmse.mean():.3f} ± {rmse.std():.3f}")


if __name__ == "__main__":
    (X_female, y_female, raw_female), (X_male, y_male, raw_male) = load_features()

    train_models("FEMALE", X_female, y_female, raw_female)
    train_models("MALE", X_male, y_male, raw_male)

    # Optional, slower diagnostics — uncomment for the honest per-fold CV number:
    cross_validate("FEMALE", X_female, y_female, raw_female)
    cross_validate("MALE", X_male, y_male, raw_male)
