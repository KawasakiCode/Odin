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


def load_features():
    """Ratios + shape features per sex, plus the fitted shape model per sex."""
    Xf, Xm, yf, ym = split_by_gender()
    id2lm = load_landmarks()

    def augment(X, y):
        ids = [i for i in X.index if i in id2lm]
        X, y = X.loc[ids], y.loc[ids]
        raw = np.stack([id2lm[i] for i in ids]).astype(np.float64)
        M, pca = fit_shape_model(raw)
        shp = pd.DataFrame(shape_feature_matrix(raw, M, pca),
                           columns=SHAPE_NAMES, index=X.index)
        return pd.concat([X, shp], axis=1), y, (M, pca)

    Xf, yf, shape_f = augment(Xf, yf)
    Xm, ym, shape_m = augment(Xm, ym)
    return Xf, Xm, yf, ym, shape_f, shape_m


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


def train_models(label, X, y, shape_model):
    """Train and evaluate XGBoost on one gender subset, then export it."""
    print("\n" + "=" * 70)
    print(f"  {label}  ({len(X)} faces, {X.shape[1]} features)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

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
    report(f"{label} — XGBoost", xgb_model, X_test, y_test, X.columns)

    export_model(label, X, y, xgb_model.best_iteration + 1, shape_model)


def export_model(label, X, y, n_estimators, shape_model):
    """Refit XGB on all rows and save it to models/ with metadata + shape model.

    The feature column order, the target, the male-boost calibration stats, and
    the Procrustes shape model (consensus mean + PCA) are bundled with the model
    so inference reproduces the exact feature vector.
    """
    MODEL_DIR = BASE / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    slug = label.lower().replace(" ", "_")
    ref_mean, pca = shape_model

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
        "shape_ref_mean": ref_mean,
        "shape_pca": pca,
    }
    out = MODEL_DIR / f"model_{slug}.joblib"
    joblib.dump(bundle, out)
    print(f"\n{label} — saved trained model -> {out} "
          f"({len(X)} rows, {X.shape[1]} features)")


def cross_validate(label, X, y, n_splits=5):
    """K-fold CV for XGBoost (mean ± std). Optional diagnostic — slow-ish."""
    print("\n" + "#" * 70)
    print(f"  {label}  —  {n_splits}-fold cross-validation  ({len(X)} faces)")
    print("#" * 70)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = {"r2": [], "mae": [], "rmse": []}
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
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
    X_female, X_male, y_female, y_male, shape_f, shape_m = load_features()

    train_models("FEMALE", X_female, y_female, shape_f)
    train_models("MALE", X_male, y_male, shape_m)

    # Optional, slower diagnostics — uncomment to run honest CV:
    # cross_validate("FEMALE", X_female, y_female)
    # cross_validate("MALE", X_male, y_male)
