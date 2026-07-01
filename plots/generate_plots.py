"""
Generate portfolio / interview plots for Odin.

All plots are AGGREGATE statistics (no face images), safe to show/commit.
Run from the repo root with the env that has matplotlib/sklearn/xgboost:
    venv/Scripts/python.exe plots/generate_plots.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots"
CSV = ROOT / "odin_training" / "training_data_scut.csv"
MODELS = {"male": ROOT / "models" / "model_male.joblib",
          "female": ROOT / "models" / "model_female.joblib"}

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25})
NAVY, TEAL, ORANGE, GREEN, RED = "#22304a", "#2a9d8f", "#e76f51", "#4caf50", "#c1121f"

# ---- shared data -----------------------------------------------------------
df = pd.read_csv(CSV)
df["sex"] = df["Image_ID"].str[1]
FEATURES = [c for c in df.columns if c not in ("Image_ID", "Attractiveness", "sex")]


def category(name):
    if name == "averageness" or name.startswith("shape_pc"):
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
    male = [0.613, 0.749, 0.81]
    female = [0.633, 0.759, 0.81]
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


# ---- 2. predicted vs actual (honest 5-fold OOF) ----------------------------
def oof(X, y):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
        m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.7, reg_lambda=2,
                             random_state=42)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def plot_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, sex, color in zip(axes, ("male", "female"), (NAVY, TEAL)):
        sub = df[df["sex"] == sex.upper()[0]]
        X = sub[FEATURES].fillna(sub[FEATURES].mean()).values
        y = sub["Attractiveness"].values
        p = oof(X, y)
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


# ---- 5. Procrustes shape ΔR² (multi-seed) ----------------------------------
def plot_procrustes():
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(["Male", "Female"], [0.013, 0.013], yerr=[0.005, 0.005],
           capsize=8, color=[NAVY, TEAL], width=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ R²  (shape features added, over 10 seeds)")
    ax.set_ylim(-0.005, 0.025)
    ax.set_title("Procrustes shape features: small but real\n"
                 "+0.013 ± 0.005 R² — sign never flips across seeds")
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
