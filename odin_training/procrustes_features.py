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

# Force strictly single-threaded execution. These cap the BLAS/OpenMP pools that
# numpy/scipy/sklearn use for the SVDs in GPA and PCA, and MUST be set before
# numpy is imported. XGBoost is pinned separately via n_jobs=1. Slower, but never
# oversubscribes the machine.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
import joblib
from shape_utils import load_landmarks
from Odin.Face_analysis.Ratios.shape import align_to_reference
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import xgboost as xgb

N_PCA = 15
RS = [42, 45, 33, 12, 63, 67, 83, 23, 98, 10]

# The colour-unevenness / spot descriptors whose marginal value we test.
SKIN_FEATURES = ["skin_a_std", "skin_b_std", "skin_spot_burden"]


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

def _fit_xgb(Xtr, ytr, Xte, rs):
    """Fit one XGB (train.py params, early stopping) and predict Xte."""
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=rs)
    # Strictly single-threaded (n_jobs=1) so it never oversubscribes the machine
    # (train.py's own params are otherwise matched).
    m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.7, reg_lambda=2,
                         early_stopping_rounds=50, random_state=rs, n_jobs=1)
    m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    return m.predict(Xte)


def run_skin_dr2():
    """
    Marginal ΔR² of the skin colour-unevenness / spot features.

    For each of the RS seeds, two full models (ratios + averageness + shape-PCA)
    are trained with 5-fold OOF — one WITH the three skin features, one WITHOUT —
    sharing the same folds and the same per-fold shape model, so the only
    difference is those three columns. Reports mean ± std (ddof=1, sample) of the
    paired ΔR² over the 10 seeds, per sex.
    """
    # Read each archive member ONCE — npz[...] is a lazy loader that re-reads and
    # re-decompresses the whole array on every access, so indexing it inside the
    # loop would reload ~40 MB per iteration and blow up memory.
    npz = np.load("landmarks_scut.npz", allow_pickle=True)
    ids = npz["ids"]
    lm = npz["landmarks"]  # (N, 478, 2), loaded once
    id2lm = {i: lm[k] for k, i in tqdm(enumerate(ids), total=len(ids),
                                       desc="index landmarks")}

    df = pd.read_csv("training_data_scut.csv")
    df = df[df["Image_ID"].isin(id2lm)].reset_index(drop=True)
    df["sex"] = df["Image_ID"].str[1]

    fn_all = [c for c in df.columns if c not in ("Image_ID", "Attractiveness", "sex")]
    fn_no = [c for c in fn_all if c not in SKIN_FEATURES]
    missing = [c for c in SKIN_FEATURES if c not in fn_all]
    if missing:
        raise SystemExit(f"CSV is missing {missing} — delete/regenerate "
                         "training_data_scut.csv so the skin features are present.")

    # Precompute per sex once (GPA alignment is label-free, seed-independent).
    data = {}
    for sex, name in tqdm([("F", "FEMALE"), ("M", "MALE")],
                          desc="GPA align per sex"):
        sub = df[df["sex"] == sex].reset_index(drop=True)
        raw = np.stack([id2lm[i] for i in sub["Image_ID"]])
        data[name] = dict(
            y=sub["Attractiveness"].values,
            Xall=sub[fn_all].fillna(sub[fn_all].mean()).values,
            Xno=sub[fn_no].fillna(sub[fn_no].mean()).values,
            aligned=gpa(raw).reshape(len(sub), -1),
        )

    withs = {"FEMALE": [], "MALE": []}
    withouts = {"FEMALE": [], "MALE": []}
    deltas = {"FEMALE": [], "MALE": []}

    for rs in tqdm(RS, desc="seeds"):
        for name in tqdm(("FEMALE", "MALE"), desc=f"seed {rs}", leave=False):
            d = data[name]
            y, aligned = d["y"], d["aligned"]
            oof_w, oof_n = np.zeros(len(y)), np.zeros(len(y))
            for tr, te in tqdm(KFold(5, shuffle=True, random_state=rs).split(np.arange(len(y))),
                               total=5, desc=f"{name} folds", leave=False):
                mean_tr = aligned[tr].mean(axis=0)
                pca = PCA(n_components=N_PCA, random_state=rs).fit(aligned[tr])

                def aug(X, idx):
                    avg = np.linalg.norm(aligned[idx] - mean_tr, axis=1, keepdims=True)
                    return np.hstack([X[idx], avg, pca.transform(aligned[idx])])

                oof_w[te] = _fit_xgb(aug(d["Xall"], tr), y[tr], aug(d["Xall"], te), rs)
                oof_n[te] = _fit_xgb(aug(d["Xno"], tr), y[tr], aug(d["Xno"], te), rs)

            r2w, r2n = r2_score(y, oof_w), r2_score(y, oof_n)
            withs[name].append(r2w)
            withouts[name].append(r2n)
            deltas[name].append(r2w - r2n)
        print(f"  seed {rs}: "
              f"F Δ={deltas['FEMALE'][-1]:+.4f}  M Δ={deltas['MALE'][-1]:+.4f}")

    for name in ("FEMALE", "MALE"):
        w, n, dl = (np.array(withs[name]), np.array(withouts[name]),
                    np.array(deltas[name]))
        print(f"\n=== {name}: skin-feature ΔR² over {len(RS)} seeds ===")
        print(f"  without skin : R2 = {n.mean():.3f} ± {n.std(ddof=1):.3f}")
        print(f"  with skin    : R2 = {w.mean():.3f} ± {w.std(ddof=1):.3f}")
        print(f"  ΔR²          : {dl.mean():+.4f} ± {dl.std(ddof=1):.4f}")


def run_shape_compare(k_pca=15, k_pls=15, seeds=RS):
    """
    PCA vs PLS shape projection, honest 10-seed 5-fold OOF, per sex.

    Three models are compared: ratios only, ratios + PCA-shape, ratios + PLS-shape
    (both projections carry the same averageness feature, so the only difference is
    HOW the shape axes are chosen). Both the PCA and the PLS basis are refit on each
    fold's TRAIN split — mandatory for PLS, which sees y — so nothing leaks. Prints
    mean ± std (ddof=1) R² and the paired PLS − PCA gap. Pass seeds=[42] for a quick
    single-seed smoke test.
    """
    npz = np.load("landmarks_scut.npz", allow_pickle=True)
    ids, lm = npz["ids"], npz["landmarks"]
    id2lm = {i: lm[k] for k, i in enumerate(ids)}

    df = pd.read_csv("training_data_scut.csv")
    df = df[df["Image_ID"].isin(id2lm)].reset_index(drop=True)
    df["sex"] = df["Image_ID"].str[1]
    fn = [c for c in df.columns if c not in ("Image_ID", "Attractiveness", "sex")]

    data = {}
    for sex, name in [("F", "FEMALE"), ("M", "MALE")]:
        sub = df[df["sex"] == sex].reset_index(drop=True)
        raw = np.stack([id2lm[i] for i in sub["Image_ID"]])
        data[name] = dict(
            y=sub["Attractiveness"].values,
            Xr=sub[fn].fillna(sub[fn].mean()).values,
            aligned=gpa(raw).reshape(len(sub), -1),
        )

    res = {n: {"ratio": [], "pca": [], "pls": []} for n in ("FEMALE", "MALE")}
    for rs in tqdm(seeds, desc="seeds"):
        for name in ("FEMALE", "MALE"):
            d = data[name]
            y, Xr, aligned = d["y"], d["Xr"], d["aligned"]
            oof = {k: np.zeros(len(y)) for k in ("ratio", "pca", "pls")}
            for tr, te in tqdm(KFold(5, shuffle=True, random_state=rs).split(np.arange(len(y))),
                               total=5, desc=f"{name} folds", leave=False):
                mean_tr = aligned[tr].mean(axis=0)
                pca = PCA(n_components=k_pca, random_state=rs).fit(aligned[tr])
                pls = PLSRegression(n_components=k_pls).fit(aligned[tr], y[tr])

                def build(proj, idx):
                    avg = np.linalg.norm(aligned[idx] - mean_tr, axis=1, keepdims=True)
                    return np.hstack([Xr[idx], avg, proj.transform(aligned[idx])])

                oof["ratio"][te] = _fit_xgb(Xr[tr], y[tr], Xr[te], rs)
                oof["pca"][te] = _fit_xgb(build(pca, tr), y[tr], build(pca, te), rs)
                oof["pls"][te] = _fit_xgb(build(pls, tr), y[tr], build(pls, te), rs)

            for key in ("ratio", "pca", "pls"):
                res[name][key].append(r2_score(y, oof[key]))

    for name in ("FEMALE", "MALE"):
        r = {k: np.array(v) for k, v in res[name].items()}
        d_pca, d_pls = r["pca"] - r["ratio"], r["pls"] - r["ratio"]
        gap = r["pls"] - r["pca"]
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
        print(f"\n=== {name}  (k_pca={k_pca}, k_pls={k_pls}, {len(seeds)} seeds) ===")
        print(f"  ratios only : R2 = {r['ratio'].mean():.3f} +/- {sd(r['ratio']):.3f}")
        print(f"  + PCA shape : R2 = {r['pca'].mean():.3f} +/- {sd(r['pca']):.3f}"
              f"   d vs ratios = {d_pca.mean():+.4f}")
        print(f"  + PLS shape : R2 = {r['pls'].mean():.3f} +/- {sd(r['pls']):.3f}"
              f"   d vs ratios = {d_pls.mean():+.4f}")
        print(f"  PLS - PCA   : {gap.mean():+.4f} +/- {sd(gap):.4f}")


def _shape_oof_r2(d, rs, make_proj):
    """
    Honest 5-fold OOF R2 for ratios (+ averageness + shape projection) on one sex,
    one seed. make_proj(X_train, y_train) returns a fitted projector with .transform
    (PCA or PLS), or None for the ratios-only baseline. The projector is refit on
    each fold's TRAIN split, so nothing leaks — mandatory for PLS.
    """
    y, Xr, aligned = d["y"], d["Xr"], d["aligned"]
    oof = np.zeros(len(y))
    for tr, te in tqdm(KFold(5, shuffle=True, random_state=rs).split(np.arange(len(y))),
                       total=5, desc="folds", leave=False):
        mean_tr = aligned[tr].mean(axis=0)
        proj = make_proj(aligned[tr], y[tr])

        def build(idx):
            if proj is None:
                return Xr[idx]
            avg = np.linalg.norm(aligned[idx] - mean_tr, axis=1, keepdims=True)
            return np.hstack([Xr[idx], avg, proj.transform(aligned[idx])])

        oof[te] = _fit_xgb(build(tr), y[tr], build(te), rs)
    return r2_score(y, oof)


def run_pls_k_sweep(k_list=(6, 8, 10, 12, 15), seeds=RS, include_ref=True):
    """
    Sweep the PLS component count to find where OOF R2 stops paying.

    For each k in k_list, reports honest 10-seed 5-fold OOF R2 of ratios + PLS(k)
    shape per sex. With include_ref=True, ratios-only and PCA(15) are computed once
    as reference lines (set False to skip them when you already have those numbers).
    Because PLS orders axes by relevance, you can push k for R2 and still interpret
    only the leading axes.
    """
    npz = np.load("landmarks_scut.npz", allow_pickle=True)
    ids, lm = npz["ids"], npz["landmarks"]
    id2lm = {i: lm[k] for k, i in enumerate(ids)}

    df = pd.read_csv("training_data_scut.csv")
    df = df[df["Image_ID"].isin(id2lm)].reset_index(drop=True)
    df["sex"] = df["Image_ID"].str[1]
    fn = [c for c in df.columns if c not in ("Image_ID", "Attractiveness", "sex")]

    data = {}
    for sex, name in [("F", "FEMALE"), ("M", "MALE")]:
        sub = df[df["sex"] == sex].reset_index(drop=True)
        raw = np.stack([id2lm[i] for i in sub["Image_ID"]])
        data[name] = dict(
            y=sub["Attractiveness"].values,
            Xr=sub[fn].fillna(sub[fn].mean()).values,
            aligned=gpa(raw).reshape(len(sub), -1),
        )

    # Reference lines (ratios only, PCA(15)) — computed once, unless skipped.
    ref = {n: {"ratio": [], "pca15": []} for n in ("FEMALE", "MALE")}
    if include_ref:
        for rs in tqdm(seeds, desc="reference seeds"):
            for name in ("FEMALE", "MALE"):
                ref[name]["ratio"].append(
                    _shape_oof_r2(data[name], rs, lambda Xt, yt: None))
                ref[name]["pca15"].append(
                    _shape_oof_r2(data[name], rs,
                                  lambda Xt, yt, rs=rs: PCA(15, random_state=rs).fit(Xt)))

    # PLS sweep.
    sweep = {n: {k: [] for k in k_list} for n in ("FEMALE", "MALE")}
    for k in tqdm(k_list, desc="k_pls"):
        for rs in tqdm(seeds, desc=f"k={k} seeds", leave=False):
            for name in ("FEMALE", "MALE"):
                sweep[name][k].append(
                    _shape_oof_r2(data[name], rs,
                                  lambda Xt, yt, k=k: PLSRegression(k).fit(Xt, yt)))

    sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
    for name in ("FEMALE", "MALE"):
        print(f"\n=== {name}  ({len(seeds)} seeds) ===")
        rt = np.array(ref[name]["ratio"]) if include_ref else None
        if include_ref:
            pc = np.array(ref[name]["pca15"])
            print(f"  ratios only : R2 = {rt.mean():.3f} +/- {sd(rt):.3f}")
            print(f"  + PCA(15)   : R2 = {pc.mean():.3f} +/- {sd(pc):.3f}"
                  f"   d = {(pc - rt).mean():+.4f}")
        for k in k_list:
            v = np.array(sweep[name][k])
            dtxt = f"   d = {(v - rt).mean():+.4f}" if include_ref else ""
            print(f"  + PLS({k:>2})   : R2 = {v.mean():.3f} +/- {sd(v):.3f}{dtxt}")


def _sex_shape_scores(sex):
    """
    Load the (retrained) bundle and this sex's training faces, and return
    (bundle, df, flat, scores, mean_flat): the GPA-aligned flattened shapes, the
    PLS scores per face (matching shape_pls_01..N), and the mean shape.
    """
    bundle = joblib.load(MODEL_PATHS[sex])
    M = bundle["shape_ref_mean"]
    proj = bundle.get("shape_pls") or bundle["shape_pca"]   # PLS now; PCA = legacy

    id2lm = load_landmarks()
    df = pd.read_csv(Path(__file__).resolve().parent / "training_data_scut.csv")
    letter = "M" if sex == "male" else "F"           # bundle uses word, CSV uses letter
    df = df[df["Image_ID"].str[1] == letter]
    df = df[df["Image_ID"].isin(id2lm)].reset_index(drop=True)

    ids = df["Image_ID"].tolist()
    raw = np.stack([id2lm[i] for i in ids]).astype(np.float64)
    flat = np.stack([align_to_reference(c, M) for c in raw]).reshape(len(raw), -1)
    scores = proj.transform(flat)
    return bundle, df, flat, scores, flat.mean(axis=0)


def _axis_deformation(flat, scores, mean_flat, i, n_sigma=2.0):
    """
    Empirical deformation for shape axis i: the OLS slope of each landmark
    coordinate on score i — how the shape moves per unit of that score. Method-
    agnostic (works for PLS or PCA) and stays in original coordinate units, so no
    projector-specific scaling is needed. Returns (minus, plus) as (P, 2) arrays
    at -/+ n_sigma standard deviations of the score.
    """
    t = scores[:, i]
    tc = t - t.mean()
    direction = (flat - mean_flat).T @ tc / (tc @ tc)      # (2P,)
    step = n_sigma * t.std() * direction
    return (mean_flat - step).reshape(-1, 2), (mean_flat + step).reshape(-1, 2)


def run_interpretability_and_plot(sex):
    """Render a -2σ/+2σ deformation plot per PLS shape axis into shape_axes_plots/."""
    out_dir = PROJECT_ROOT / "shape_axes_plots"
    out_dir.mkdir(exist_ok=True)

    _, _, flat, scores, mean_flat = _sex_shape_scores(sex)
    n = scores.shape[1]
    for i in range(n):
        minus, plus = _axis_deformation(flat, scores, mean_flat, i)

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
        ax.set_title(f"{sex}  ·  shape_pls_{i + 1:02d}")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sex}_shape_pls_{i + 1:02d}.png", dpi=130)
        plt.close(fig)

    print(f"saved {n} axis plots -> {out_dir}")


def correlate_axes_with_ratios(sex):
    """
    Name each PLS shape axis objectively: for every face, correlate its axis score
    (from the BUNDLE's shape model, so it matches shape_pls_01..N) against every
    named ratio, and print the top-3 correlated ratios per axis alongside the
    model's global (gain) importance for that axis. Sorted by importance, so the
    axes the model actually pays for come first.
    """
    bundle, df, _, scores, _ = _sex_shape_scores(sex)
    imp = dict(zip(bundle["feature_names"], bundle["xgboost"].feature_importances_))

    ratio_cols = [c for c in df.columns
                  if c not in ("Image_ID", "Attractiveness", "sex")]
    R = df[ratio_cols].fillna(df[ratio_cols].mean()).values

    rows = []
    for k in range(scores.shape[1]):
        name = f"shape_pls_{k + 1:02d}"
        corr = sorted(
            ((c, np.corrcoef(scores[:, k], R[:, j])[0, 1])
             for j, c in enumerate(ratio_cols)),
            key=lambda t: abs(t[1]), reverse=True)
        top = ", ".join(f"{c} ({r:+.2f})" for c, r in corr[:3])
        rows.append((name, imp.get(name, 0.0), top))

    rows.sort(key=lambda r: r[1], reverse=True)   # most-important axis first
    print(f"\n=== {sex}: shape axes (by model importance) ===")
    print(f"{'axis':14} {'import':>7}   top-correlated ratios")
    for name, im, top in rows:
        print(f"{name:14} {im:7.3f}   {top}")


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

    
