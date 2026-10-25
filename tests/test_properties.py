"""
Properties the features and model are designed to have, independent of any
recorded values: ratio and shape invariances, the averageness scale, and the
male presentation boost.
"""
import numpy as np
import pytest

from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.Ratios.shape import align_to_reference, shape_feature_dict
from Odin.main import build_features, male_boost


def ratios(px):
    """The 25 geometric ratios (no appearance features: those need pixels)."""
    return build_features(extract_face_data(px), {})


def similarity(px, scale, angle_deg, dx, dy):
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = px.copy()
    out[:, :2] = scale * px[:, :2] @ R.T + [dx, dy]
    out[:, 2] = scale * px[:, 2]
    return out


def test_ratios_invariant_to_translation_and_scale(face):
    base = ratios(face["px"])
    moved = ratios(similarity(face["px"], 2.7, 0, 113, -41))
    for k, v in base.items():
        assert moved[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k


def test_shape_features_invariant_to_similarity_transforms(face, models):
    model = models[face["record"]["sex"]]
    proj, ref = model["shape_pls"], model["shape_ref_mean"]
    base = shape_feature_dict(face["px"][:, :2], ref, proj)
    moved = shape_feature_dict(similarity(face["px"], 0.4, 23, -500, 77)[:, :2], ref, proj)
    for k, v in base.items():
        assert moved[k] == pytest.approx(v, rel=1e-7, abs=1e-9), k


def test_mean_shape_has_zero_averageness(models):
    for model in models.values():
        ref = model["shape_ref_mean"]
        aligned = align_to_reference(ref * 900 + [300, 200], ref)
        assert np.linalg.norm(aligned - ref) < 1e-12


def test_averageness_quantiles_are_a_valid_distribution(models):
    for model in models.values():
        q = np.asarray(model["averageness_quantiles"])
        assert len(q) == 101 and (np.diff(q) >= 0).all() and q[0] >= 0


@pytest.mark.parametrize("sex", ["male", "female"])
def test_male_boost(models, sex):
    model = models[sex]
    mean, cap = model["xgb_pred_mean"], model["xgb_pred_max"]
    below = np.linspace(1, mean, 20)
    above = np.linspace(mean + 1e-6, 10, 50)
    # unchanged at/below the mean, monotone and capped above it
    assert [male_boost(s, model) for s in below] == pytest.approx(below)
    boosted = np.array([male_boost(s, model) for s in above])
    assert (np.diff(boosted) >= -1e-12).all()
    assert boosted.max() <= cap + 1e-12
    # never lowers a score, except where the raw score itself exceeds the cap
    assert ((boosted >= above - 1e-12) | (boosted == cap)).all()
    # bundles without calibration stats are left alone
    assert male_boost(7.0, {}) == 7.0
