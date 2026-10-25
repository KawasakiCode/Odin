"""
The browser build serves models exported from the .joblib bundles. If someone
retrains without re-running odin_web/export_model.py, the site would silently
serve an old model — this catches it.
"""
import json
import sys

import pytest

from conftest import ROOT, needs_landmarker

sys.path.insert(0, str(ROOT / "odin_web"))
import export_model  # noqa: E402


@needs_landmarker   # the export also copies face_landmarker.task
def test_committed_web_models_match_joblib_bundles(tmp_path):
    export_model.main(["--out", str(tmp_path)])
    for sex in ("male", "female"):
        fresh = json.loads((tmp_path / f"model_{sex}.json").read_text())
        committed = json.loads((ROOT / "odin_web" / "public" / "models" / f"model_{sex}.json").read_text())
        assert fresh == committed, (
            f"odin_web/public/models/model_{sex}.json is stale: run odin_web/export_model.py")


@pytest.mark.parametrize("sex", ["male", "female"])
def test_web_model_has_named_features_only(sex):
    m = json.loads((ROOT / "odin_web" / "public" / "models" / f"model_{sex}.json").read_text())
    assert set(m["shape"]["components"]) == {f for f in m["featureNames"] if f.startswith("shape_pls_")}
    assert all(len(t["left"]) == len(t["value"]) == len(t["cover"]) for t in m["trees"])
