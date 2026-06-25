"""
Training-side helpers for the Procrustes shape features.

Fits the consensus mean shape (GPA) and the shape-PCA basis on a set of faces,
and turns faces into [averageness, shape_pc_01..N]. Uses the SAME per-face
transform as inference (Odin.Face_analysis.Ratios.shape) so the features match
exactly between training and prediction.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Odin.Face_analysis.Ratios.shape import _center_scale, align_to_reference

N_PCA = 15
RANDOM_STATE = 42
SHAPE_NAMES = ["averageness"] + [f"shape_pc_{i:02d}" for i in range(1, N_PCA + 1)]
# Anchor the cache to this file's folder so it resolves from any CWD.
LANDMARK_CACHE = str(Path(__file__).resolve().parent / "landmarks_scut.npz")


def load_landmarks(path=LANDMARK_CACHE):
    """{Image_ID: (P, 2) array}. Raises if the cache hasn't been built yet."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found — run `python extract_landmarks.py` first to "
            "build the landmark cache the shape features need."
        )
    npz = np.load(path, allow_pickle=True)
    return dict(zip(list(npz["ids"]), npz["landmarks"]))


def gpa(raw, iters=5):
    """Generalized Procrustes alignment -> consensus mean shape (P, 2)."""
    X = np.stack([_center_scale(c) for c in raw])
    M = X[0].copy()
    for _ in range(iters):
        for i in range(len(X)):
            R, _ = orthogonal_procrustes(X[i], M)
            X[i] = X[i] @ R
        M = X.mean(axis=0)
        M /= np.sqrt((M ** 2).sum())
    return M


def fit_shape_model(raw):
    """Fit (consensus mean, PCA) on a set of (P, 2) configs."""
    M = gpa(raw)
    flat = np.stack([align_to_reference(c, M) for c in raw]).reshape(len(raw), -1)
    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE).fit(flat)
    return M, pca


def shape_feature_matrix(raw, M, pca):
    """[averageness, shape_pc_01..N] for each (P, 2) config, as an (n, 16) array."""
    flat = np.stack([align_to_reference(c, M) for c in raw]).reshape(len(raw), -1)
    avg = np.linalg.norm(flat - M.reshape(1, -1), axis=1)
    return np.column_stack([avg, pca.transform(flat)])
