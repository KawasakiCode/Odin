"""
Procrustes shape features (averageness + shape-PCA).

At training time a consensus mean shape and a PCA basis are fit over all faces
and stored in the model bundle. At inference a single new face is superimposed
on that stored mean (translate -> scale -> rotate) and turned into:
  - averageness: Procrustes distance of the aligned face to the mean shape,
  - shape_pc_01..N: its projection onto the stored shape-PCA basis.

The transform here must match how the training features were built (same
centre/scale, same row-major flatten of the (P, 2) coordinates).
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes


def _center_scale(config):
    """Centre at centroid and scale to unit centroid size."""
    c = np.asarray(config, dtype=np.float64)
    c = c - c.mean(axis=0)
    size = np.sqrt((c ** 2).sum())
    return c / size if size else c


def align_to_reference(config_xy, ref_mean):
    """Procrustes-superimpose a (P,2) config onto the reference mean shape."""
    c = _center_scale(config_xy)
    R, _ = orthogonal_procrustes(c, ref_mean)
    return c @ R


def shape_feature_dict(config_xy, ref_mean, pca):
    """
    {averageness, shape_pc_01..N} for one face.

    config_xy: (P, 2) landmark coords (pixel space; scale is removed here).
    ref_mean:  (P, 2) consensus mean shape stored in the bundle.
    pca:       fitted sklearn PCA over row-major-flattened aligned shapes.
    """
    aligned = align_to_reference(config_xy, ref_mean)
    averageness = float(np.linalg.norm(aligned - ref_mean))
    pcs = pca.transform(aligned.reshape(1, -1))[0]
    feats = {"averageness": averageness}
    for i, v in enumerate(pcs, start=1):
        feats[f"shape_pc_{i:02d}"] = float(v)
    return feats
