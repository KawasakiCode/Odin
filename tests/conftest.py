"""
Shared fixtures. The reference faces are the committed CI fixtures in
odin_web/parity/fixtures_ci/: public-domain AI-generated sample faces (plus
mirrored / resized / rotated / cropped variants) with the Python pipeline's
output recorded by odin_web/parity/make_fixtures.py.
"""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "odin_web" / "parity" / "fixtures_ci"
LANDMARKER = ROOT / "models" / "face_landmarker.task"

_fixtures = json.loads((FIXTURES / "fixtures.json").read_text())


def fixture_ids():
    return [f["id"] for f in _fixtures]


@pytest.fixture(scope="session")
def models():
    return {s: joblib.load(ROOT / "models" / f"model_{s}.joblib") for s in ("male", "female")}


@pytest.fixture(params=_fixtures, ids=fixture_ids())
def face(request):
    """One reference face: its record, image (BGR) and pixel-space landmarks."""
    f = request.param
    img = cv2.imread(str(FIXTURES / f"{f['id']}.png"))
    h, w = img.shape[:2]
    px = np.array(f["landmarks"], dtype=np.float64)
    px[:, 0] *= w
    px[:, 1] *= h
    px[:, 2] *= w
    return {"record": f, "img": img, "px": px, "path": FIXTURES / f"{f['id']}.png"}


needs_landmarker = pytest.mark.skipif(
    not LANDMARKER.exists(), reason="models/face_landmarker.task not downloaded")
