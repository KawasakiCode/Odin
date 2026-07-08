"""
Generate portfolio / interview plots for Odin.

All plots are AGGREGATE statistics (no face images), safe to show/commit.
Run from the repo root with the env that has matplotlib/sklearn/xgboost:
    venv/Scripts/python.exe plots/generate_plots.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots"
CSV = ROOT / "odin_training" / "training_data_scut.csv"
MODELS = {"male": ROOT / "models" / "model_male.joblib",
          "female": ROOT / "models" / "model_female.joblib"}

# Shape features (PLS) are built at train time, not stored in the CSV, so the
# honest-OOF plot needs the same fit-per-fold helpers the trainer uses.
sys.path.insert(0, str(ROOT / "odin_training"))
from shape_utils import fit_shape_model, shape_feature_matrix, load_landmarks

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25})
NAVY, TEAL, ORANGE, GREEN, RED = "#22304a", "#2a9d8f", "#e76f51", "#4caf50", "#c1121f"

# ---- shared data -----------------------------------------------------------
df = pd.read_csv(CSV)
df["sex"] = df["Image_ID"].str[1]
FEATURES = [c for c in df.columns if c not in ("Image_ID", "Attractiveness", "sex")]


def category(name):
    if name == "averageness" or name.startswith(("shape_pls", "shape_pc")):
        return "shape"
    if (name.startswith(("skin_", "lips_", "eye_")) or "contrast" in name
            or name == "skin_texture"):
        return "appearance"
    return "geometry"


# ---- 1. ceiling comparison -------------------------------------------------
def plot_ceiling():
    labels = ["Hand-crafted\nfeatures\n(this project)",
              "Frozen FaceNet-512\nembedding + Ridge",
              "End-to-end CNN\n(SCUT benchmark)"]
    male = [0.657, 0.749, 0.81]
    female = [0.666, 0.759, 0.81]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, male, w, label="Male", color=NAVY)
    ax.bar(x + w/2, female, w, label="Female", color=TEAL)
    for i, (m, f) in enumerate(zip(male, female)):
        ax.text(i - w/2, m + .01, f"{m:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, f + .01, f"{f:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Cross-validated R²"); ax.set_ylim(0, 0.95)
    ax.set_title("How far interpretable geometry is from the ceiling\n"
                 "(same 5-fold CV protocol; CNN = Liang et al. 2018, r≈0.90→R²≈0.81)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "1_ceiling_comparison.png"); plt.close(fig)


# ---- 2. predicted vs actual (honest 5-fold OOF, full PLS model) ------------
def oof(Xr, y, raw):
    """5-fold OOF for the real model, matching train.py's cross_validate: ratios +
    PLS shape refit on each train fold (so shape never sees the test labels), and
    XGB early-stopped on a validation split carved from train."""
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(Xr):
        M, proj = fit_shape_model(raw[tr], y[tr])
        Xtr = np.hstack([Xr.values[tr], shape_feature_matrix(raw[tr], M, proj)])
        Xte = np.hstack([Xr.values[te], shape_feature_matrix(raw[te], M, proj)])
        Xt, Xv, yt, yv = train_test_split(Xtr, y[tr], test_size=0.20, random_state=42)
        m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.7, reg_lambda=2,
                             early_stopping_rounds=50, random_state=42)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        pred[te] = m.predict(Xte)
    return pred


def plot_calibration():
    id2lm = load_landmarks()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, sex, color in zip(axes, ("male", "female"), (NAVY, TEAL)):
        sub = df[df["sex"] == sex.upper()[0]]
        sub = sub[sub["Image_ID"].isin(id2lm)].reset_index(drop=True)
        Xr = sub[FEATURES].fillna(sub[FEATURES].mean())
        y = sub["Attractiveness"].values
        raw = np.stack([id2lm[i] for i in sub["Image_ID"]]).astype(np.float64)
        p = oof(Xr, y, raw)
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        mae = np.mean(np.abs(y - p))
        ax.scatter(y, p, s=6, alpha=0.25, color=color)
        lims = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.plot(lims, lims, "--", color=RED, lw=1.4, label="perfect")
        ax.set_title(f"{sex.capitalize()}  ·  R²={r2:.3f}  MAE={mae:.3f}")
        ax.set_xlabel("Actual SCUT score (1–10)")
        ax.set_ylabel("Predicted (held-out, 5-fold OOF)")
        ax.legend(loc="upper left")
    fig.suptitle("Predicted vs. actual — honest out-of-fold predictions\n"
                 "(note the tail compression: extreme faces are pulled toward the mean)")
    fig.tight_layout(); fig.savefig(OUT / "2_predicted_vs_actual.png"); plt.close(fig)


# ---- 3. feature importance (grouped by category) ---------------------------
def plot_importance():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=False)
    cmap = {"geometry": NAVY, "appearance": ORANGE, "shape": GREEN}
    for ax, sex in zip(axes, ("male", "female")):
        b = joblib.load(MODELS[sex])
        imp = pd.Series(b["xgboost"].feature_importances_, index=b["feature_names"])
        imp = imp.sort_values(ascending=False).head(15)[::-1]
        colors = [cmap[category(n)] for n in imp.index]
        ax.barh(range(len(imp)), imp.values, color=colors)
        ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=9)
        ax.set_xlabel("XGBoost gain importance")
        ax.set_title(f"{sex.capitalize()} — top 15 features")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    fig.legend(handles, cmap.keys(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("What the model actually leans on", y=1.04)
    fig.tight_layout(); fig.savefig(OUT / "3_feature_importance.png",
                                    bbox_inches="tight"); plt.close(fig)


# ---- 4. cross-population transfer ------------------------------------------
def plot_transfer():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(["Male", "Female"], [0.04, 0.35], color=[RED, ORANGE], width=0.5)
    ax.axhline(0.76, ls="--", color=NAVY, lw=1.3)
    ax.text(1.35, 0.77, "held-out SCUT (male) ≈ 0.76", color=NAVY, fontsize=9,
            ha="right")
    for i, v in enumerate([0.04, 0.35]):
        ax.text(i, v + .01, f"{v:.2f}", ha="center")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_ylim(0, 0.85)
    ax.set_title("Male 'attractiveness' does NOT transfer across rater populations\n"
                 "(model trained on SCUT, tested on Chicago Face Database)")
    fig.tight_layout(); fig.savefig(OUT / "4_cross_population_transfer.png")
    plt.close(fig)


# ---- 5. Shape projection: ratios vs +PCA vs +PLS (10-seed OOF) --------------
def plot_procrustes():
    # 10-seed 5-fold OOF R² from the ablation (procrustes_features.run_shape_compare
    # / run_pls_k_sweep). PLS(25) is the deployed axis count.
    groups = ["ratios\nonly", "+ PCA(15)\nshape", "+ PLS(25)\nshape"]
    male   = [0.603, 0.631, 0.668]
    female = [0.625, 0.638, 0.669]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 5))
    ax.bar(x - w/2, male, w, label="Male", color=NAVY)
    ax.bar(x + w/2, female, w, label="Female", color=TEAL)
    for i, (m, f) in enumerate(zip(male, female)):
        ax.text(i - w/2, m + .003, f"{m:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, f + .003, f"{f:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("10-seed 5-fold OOF R²")
    ax.set_ylim(0.55, 0.70)
    ax.set_title("Supervised shape (PLS) is the biggest lever\n"
                 "PCA adds +0.01–0.03 over ratios; PLS adds +0.04–0.07 (sign never flips)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "5_procrustes_delta.png"); plt.close(fig)


# ---- 6. label distribution -------------------------------------------------
def plot_labels():
    fig, ax = plt.subplots(figsize=(8, 5))
    for sex, color, lab in (("M", NAVY, "Male"), ("F", TEAL, "Female")):
        ax.hist(df[df["sex"] == sex]["Attractiveness"], bins=40, alpha=0.55,
                color=color, label=lab)
    ax.set_xlabel("SCUT attractiveness score (rescaled 1–10)")
    ax.set_ylabel("Number of faces")
    ax.set_title("Label distribution — SCUT-FBP5500 (2,750 faces per sex)\n"
                 "~60 predominantly East-Asian raters, averaged")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "6_label_distribution.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plot_ceiling();      print("1/6 ceiling")
    plot_calibration();  print("2/6 predicted vs actual")
    plot_importance();   print("3/6 feature importance")
    plot_transfer();     print("4/6 cross-population transfer")
    plot_procrustes();   print("5/6 procrustes delta")
    plot_labels();       print("6/6 label distribution")
    print(f"\nDone -> {OUT}")
