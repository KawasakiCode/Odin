"""
Reference outputs from OpenCV for the operations src/core/imageOps.ts
reimplements, so the TypeScript unit tests can check them without Python.
Writes tests/opencv_cases.json (committed). Regenerate after an OpenCV upgrade:
    venv/Scripts/python.exe odin_web/tests/make_opencv_cases.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

rng = np.random.default_rng(42)
cases = {"opencv": cv2.__version__}

# Colour conversions: random colours + the corners/greys most likely to hit
# rounding edges.
cols = rng.integers(0, 256, (2000, 3)).tolist()
cols += [[v, v, v] for v in range(256)] + [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
rgb = np.array(cols, np.uint8).reshape(-1, 1, 3)
bgr = rgb[:, :, ::-1].copy()
cases["colors"] = {
    "rgb": cols,
    "lab": cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).tolist(),
    "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).reshape(-1).tolist(),
}

# fillPoly: random convex hulls and random (possibly self-intersecting) loops,
# single and multiple polygons, some partly outside the canvas.
W, H = 64, 48
polys = []
for k in range(60):
    n = int(rng.integers(3, 9))
    pts = rng.integers(-6, 70, (n, 2))
    if k % 2 == 0:
        pts = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2)
    group = [pts.astype(np.int32)]
    if k % 5 == 0:
        group.append(rng.integers(0, 60, (4, 2)).astype(np.int32))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, group, 255)
    polys.append({"polygons": [g.tolist() for g in group],
                  "pixels": np.flatnonzero(mask == 255).tolist()})
cases["fillPoly"] = {"width": W, "height": H, "cases": polys}

# Gaussian blur (float64, sigma-derived kernel, REFLECT_101) and Laplacian.
img = rng.integers(0, 256, (37, 53)).astype(np.float64)
blur = []
for sigma in (0.8, 1.0, 2.1, 6.5, 20.0):         # 20: kernel wider than the image
    out = cv2.GaussianBlur(img, (0, 0), sigma)
    blur.append({"sigma": sigma, "values": out.ravel().tolist()})
gray = rng.integers(0, 256, (29, 31)).astype(np.uint8)
cases["blur"] = {"width": 53, "height": 37, "image": img.ravel().tolist(), "cases": blur}
cases["laplacian"] = {"width": 31, "height": 29, "image": gray.ravel().tolist(),
                      "values": cv2.Laplacian(gray, cv2.CV_64F).ravel().tolist()}

out = Path(__file__).resolve().parent / "opencv_cases.json"
out.write_text(json.dumps(cases))
print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, OpenCV {cv2.__version__})")
