"""
Procrustes/shape features experiment.

Adds two kinds of holistic geometry features the 40 hand-crafted ratios miss:
  - averageness: Procrustes distance of a face's shape to the consensus (mean)
    shape — a classic, strong attractiveness predictor.
  - shape-PCA: top-K principal components of the Procrustes-aligned landmark
    configuration — the "gestalt of geometry" (how everything sits together).

Reports honest 5-fold CV R²/MAE for XGB on the 40 ratios vs 40 + shape features.
The mean shape (for averageness) and the PCA basis are fit on each fold's TRAIN
split only, so the comparison isn't leaky.
"""
import sys
from pathlib import Path

# Project root = the directory that contains the `Odin` package (one level up
# from this file). Ensure it is importable before any `from Odin...` import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATHS = {
    "male": PROJECT_ROOT / "models" / "model_male.joblib",
    "female": PROJECT_ROOT / "models" / "model_female.joblib",
}

import numpy as np
import pandas as pd
import joblib
from shape_utils import shape_feature_matrix, load_landmarks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

N_PCA = 15
RS = [42, 45, 33, 12, 63, 67, 83, 23, 98, 10]


def center_scale(X):
    """Center each config at its centroid and scale to unit centroid size."""
    X = X - X.mean(axis=1, keepdims=True)
    size = np.sqrt((X ** 2).sum(axis=(1, 2), keepdims=True))
    return X / size


def gpa(X, iters=4):
    """Generalized Procrustes alignment. X: (n, p, 2) -> aligned (n, p, 2)."""
    Xc = center_scale(X.astype(np.float64))
    mean = Xc[0].copy()
    for _ in range(iters):
        for i in range(len(Xc)):
            R, _ = orthogonal_procrustes(Xc[i], mean)
            Xc[i] = Xc[i] @ R
        mean = Xc.mean(axis=0)
        mean /= np.sqrt((mean ** 2).sum())
    return Xc  # aligned configs (rotations to a stable consensus)


def xgb_oof(X, y, rs):
    """5-fold OOF predictions with early stopping (matches train.py)."""
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=rs).split(X):
        Xtr, ytr = X[tr], y[tr]
        Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=rs)
        m = xgb.XGBRegressor(n_estimators= 1000, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.7, reg_lambda=2,
                             early_stopping_rounds=50, random_state = rs)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        oof[te] = m.predict(X[te])
    return oof

def interprete_shape_axes(index, bundle):
    pca = bundle["shape_pca"]
    sd = pca.explained_variance_[index] ** 0.5
    comp = pca.components_[index]

    shape_minus = (pca.mean_ - 2 * sd * comp).reshape(478,2)
    shape_plus = (pca.mean_ + 2 * sd * comp).reshape(478,2)

    return shape_minus, shape_plus

def run_interpretability_and_plot(sex):
    """Render a -2σ/+2σ deformation plot per shape axis into shape_axes_plots/."""
    out_dir = PROJECT_ROOT / "shape_axes_plots"
    out_dir.mkdir(exist_ok=True)

    bundle = joblib.load(MODEL_PATHS[sex])
    n = bundle["shape_pca"].n_components_
    for i in range(n):
        minus, plus = interprete_shape_axes(i, bundle)

        fig, ax = plt.subplots(figsize=(4.2, 4.8))
        ax.scatter(minus[:, 0], minus[:, 1], s=6, color="#2a6fdb", label="−2σ")
        ax.scatter(plus[:, 0], plus[:, 1], s=6, color="#e4572e", label="+2σ")
        # Arrows from -2σ to +2σ show how each landmark moves along this axis.
        for a, b in zip(minus, plus):
            ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=0.4, alpha=0.5))
        ax.set_aspect("equal")
        ax.invert_yaxis()          # image y grows downward -> flip so the face is upright
        ax.axis("off")
        ax.set_title(f"{sex}  ·  shape_pc_{i + 1:02d}")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sex}_shape_pc_{i + 1:02d}.png", dpi=130)
        plt.close(fig)

    print(f"saved {n} axis plots -> {out_dir}")


def correlate_axes_with_ratios(sex):
    """
    Name each shape axis objectively: for every face, correlate its PC score
    (from the BUNDLE's shape model, so it matches shape_pc_01..15) against every
    named ratio, and print the top-3 correlated ratios per axis alongside the
    model's global (gain) importance for that axis. Sorted by importance, so the
    axes the model actually pays for come first.
    """
    bundle = joblib.load(MODEL_PATHS[sex])
    M, pca = bundle["shape_ref_mean"], bundle["shape_pca"]
    imp = dict(zip(bundle["feature_names"], bundle["xgboost"].feature_importances_))

    id2lm = load_landmarks()
    df = pd.read_csv(Path(__file__).resolve().parent / "training_data_scut.csv")
    letter = "M" if sex == "male" else "F"           # bundle uses word, CSV uses letter
    df = df[df["Image_ID"].str[1] == letter]
    df = df[df["Image_ID"].isin(id2lm)].reset_index(drop=True)

    ids = df["Image_ID"].tolist()
    raw = np.stack([id2lm[i] for i in ids]).astype(np.float64)
    pc_scores = shape_feature_matrix(raw, M, pca)[:, 1:]   # drop averageness -> (n, 15)

    ratio_cols = [c for c in df.columns
                  if c not in ("Image_ID", "Attractiveness", "sex")]
    R = df[ratio_cols].fillna(df[ratio_cols].mean()).values

    rows = []
    for k in range(pc_scores.shape[1]):
        name = f"shape_pc_{k + 1:02d}"
        corr = sorted(
            ((c, np.corrcoef(pc_scores[:, k], R[:, j])[0, 1])
             for j, c in enumerate(ratio_cols)),
            key=lambda t: abs(t[1]), reverse=True)
        top = ", ".join(f"{c} ({r:+.2f})" for c, r in corr[:3])
        rows.append((name, imp.get(name, 0.0), top))

    rows.sort(key=lambda r: r[1], reverse=True)   # most-important axis first
    print(f"\n=== {sex}: shape axes (by model importance) ===")
    print(f"{'axis':12} {'import':>7}   top-correlated ratios")
    for name, im, top in rows:
        print(f"{name:12} {im:7.3f}   {top}")


def main(rs):
    npz = np.load("landmarks_scut.npz", allow_pickle=True)
    lm_ids = list(npz["ids"])
    lm = npz["landmarks"]  # (N, 478, 2)
    id2lm = {i: lm[k] for k, i in enumerate(lm_ids)}

    df = pd.read_csv("training_data_scut.csv")
    df = df[df["Image_ID"].isin(id2lm)].reset_index(drop=True)
    df["sex"] = df["Image_ID"].str[1]
    fn = [c for c in df.columns if c not in ("Image_ID", "Attractiveness", "sex")]

    male_stats = []
    female_stats = []
    for sex, name in [("F", "FEMALE"), ("M", "MALE")]:
        sub = df[df["sex"] == sex].reset_index(drop=True)
        y = sub["Attractiveness"].values
        Xratio = sub[fn].fillna(sub[fn].mean()).values

        # GPA align this sex's faces once (pose normalisation, label-free).
        raw = np.stack([id2lm[i] for i in sub["Image_ID"]])
        aligned = gpa(raw).reshape(len(sub), -1)  # (n, 956)

        # Baseline: ratios only.
        base = xgb_oof(Xratio, y, rs)

        # Augmented: ratios + averageness + shape-PCA, fit per fold.
        oof = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=rs).split(Xratio):
            mean_tr = aligned[tr].mean(axis=0)
            pca = PCA(n_components=N_PCA, random_state=rs).fit(aligned[tr])

            def feats(idx):
                avg = np.linalg.norm(aligned[idx] - mean_tr, axis=1, keepdims=True)
                pcs = pca.transform(aligned[idx])
                return np.hstack([Xratio[idx], avg, pcs])

            Xtr_aug, Xte_aug = feats(tr), feats(te)
            ytr = y[tr]
            Xt, Xv, yt, yv = train_test_split(Xtr_aug, ytr, test_size=0.2, random_state=rs)
            m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.7, reg_lambda=2,
                                 early_stopping_rounds=50, random_state=rs)
            m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            oof[te] = m.predict(Xte_aug)

        def rep(p):
            return (r2_score(y, p), mean_absolute_error(y, p),
                    mean_squared_error(y, p) ** 0.5)

        r2b, maeb, rmseb = rep(base)
        r2a, maea, rmsea = rep(oof)
        print(f"\n=== {name} (n={len(sub)}) ===")
        print(f"  ratios only      : R2={r2b:.3f}  MAE={maeb:.3f}  RMSE={rmseb:.3f}")
        print(f"  + shape features : R2={r2a:.3f}  MAE={maea:.3f}  RMSE={rmsea:.3f}")
        print(f"  delta            : R2 {r2a - r2b:+.3f}  MAE {maea - maeb:+.3f}")
        if sex == "M":
            male_stats.extend((r2b, maeb, rmseb, r2a, maea, rmsea))
        else: 
            female_stats.extend((r2b, maeb, rmseb, r2a, maea, rmsea))
    return male_stats, female_stats


if __name__ == "__main__":
    run_interpretability_and_plot("male")
    run_interpretability_and_plot("female")

    # msr2b = 0 
    # msmaeb = 0
    # msrmseb = 0
    # msr2a = 0
    # msmaea = 0
    # msrmsea = 0
    # fsr2b = 0 
    # fsmaeb = 0
    # fsrmseb = 0
    # fsr2a = 0
    # fsmaea = 0
    # fsrmsea = 0
    # mdr_s = []
    # fdr_s = []
    # male_stats = []
    # female_stats = []
    # for rs in RS:
    #     male_stats, female_stats = main(rs)
    #     msr2b += male_stats[0]
    #     msmaeb += male_stats[1]
    #     msrmseb += male_stats[2]
    #     msr2a += male_stats[3]
    #     msmaea += male_stats[4]
    #     msrmsea += male_stats[5]
    #     mdr_s.append(male_stats[0] - male_stats[3])

    #     fsr2b += female_stats[0]
    #     fsmaeb += female_stats[1]
    #     fsrmseb += female_stats[2]
    #     fsr2a += female_stats[3]
    #     fsmaea += female_stats[4]
    #     fsrmsea += female_stats[5]
    #     fdr_s.append(female_stats[0] - female_stats[3])
    
    # mstd = np.std(mdr_s, ddof=1) # used ddof = 1 because 10 seeds are NOT the entire population of seeds
    # fstd = np.std(fdr_s, ddof=1)

    # print("MALE")
    # print(f"  ratios only      : R2={msr2b/len(RS):.3f}   MAE={msmaeb/len(RS):.3f}  RMSE={msrmseb/len(RS):.3f}")
    # print(f"  + shape features : R2={msr2a/len(RS):.3f}  MAE={msmaea/len(RS):.3f}  RMSE={msrmsea/len(RS):.3f}")
    # print(f"  delta            : R2 {msr2a/len(RS) - msr2b/len(RS):+.3f}± {mstd:.3f} MAE {msmaea/len(RS) - msmaeb/len(RS):+.3f}")

    # print("FEMALE")
    # print(f"  ratios only      : R2={fsr2b/len(RS):.3f}   MAE={fsmaeb/len(RS):.3f}  RMSE={fsrmseb/len(RS):.3f}")
    # print(f"  + shape features : R2={fsr2a/len(RS):.3f}  MAE={fsmaea/len(RS):.3f}  RMSE={fsrmsea/len(RS):.3f}")
    # print(f"  delta            : R2 {fsr2a/len(RS) - fsr2b/len(RS):+.3f}± {fstd:.3f} MAE {fsmaea/len(RS) - fsmaeb/len(RS):+.3f}")

    
