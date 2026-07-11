"""
Run the three SCUT-FBP5500 benchmark CNNs (Liang et al. 2018) on a face, via
OpenCV's Caffe importer — no Caffe install needed. These are the models that
reach ~0.81 R² on the dataset; the UI shows them next to Odin's score.

Each net is a fine-tuned ImageNet backbone with a single regression output on
the SCUT 1-5 beauty scale. We crop the face from the landmarks (the nets expect
a face, like the SCUT training crops), subtract the ImageNet BGR mean, and
rescale the 1-5 output to Odin's 1-10 for a like-for-like display.
"""
from pathlib import Path

import cv2
import numpy as np

DIR = Path(__file__).resolve().parent

# name -> (deploy prototxt, caffemodel, input size, output blob)
MODELS = {
    "alexnet":   ("alexnet_deploy.prototxt",   "alexnet.caffemodel",   227, "ip1"),
    "resnet18":  ("resnet18_deploy.prototxt",  "resnet18.caffemodel",  224, "feat1"),
    "resnext50": ("resnext50_deploy.prototxt", "resnext50.caffemodel", 224, "feat1"),
}
# ImageNet-style normalisation: subtract the ImageNet BGR mean, then scale by
# 1/255. Verified empirically — this is the only preprocessing under which all
# three nets agree closely on the same face (they were trained identically), and
# it puts outputs on the expected SCUT 1-5 range.
MEAN = (104.0, 117.0, 123.0)
SCALE = 1.0 / 255.0

_nets = {}


def _net(name):
    if name not in _nets:
        proto, weights, _, _ = MODELS[name]
        _nets[name] = cv2.dnn.readNetFromCaffe(str(DIR / proto), str(DIR / weights))
    return _nets[name]


def _face_crop(img_bgr, landmarks_px, margin=0.45):
    """Square crop around the landmark bounding box (+margin for hair/chin, like
    the SCUT face images), replicate-padded if it runs off the image."""
    xs, ys = landmarks_px[:, 0], landmarks_px[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin)
    L, T = int(round(cx - side / 2)), int(round(cy - side / 2))
    R, B = int(round(cx + side / 2)), int(round(cy + side / 2))
    H, W = img_bgr.shape[:2]
    padL, padT, padR, padB = max(0, -L), max(0, -T), max(0, R - W), max(0, B - H)
    crop = img_bgr[max(0, T):min(H, B), max(0, L):min(W, R)]
    if padL or padT or padR or padB:
        crop = cv2.copyMakeBorder(crop, padT, padB, padL, padR, cv2.BORDER_REPLICATE)
    return crop


def score_all(img_bgr, landmarks_px):
    """{'alexnet','resnet18','resnext50'} beauty scores rescaled to Odin's 1-10."""
    crop = _face_crop(img_bgr, landmarks_px)
    out = {}
    for name, (_, _, size, blob) in MODELS.items():
        try:
            net = _net(name)
            b = cv2.dnn.blobFromImage(crop, scalefactor=SCALE, size=(size, size),
                                      mean=MEAN, swapRB=False, crop=False)
            net.setInput(b)
            raw = float(net.forward(blob).ravel()[0])   # ~1-5 (SCUT scale)
            out[name] = round(raw * 2.0, 2)             # -> 1-10 like Odin
        except Exception:
            out[name] = None
    return out
