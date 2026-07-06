"""
Appearance-sampling regions built from MediaPipe FaceMesh landmarks.

These polygons mark the areas we will later sample for colour (lips / iris /
skin hex + contrast) and skin texture (Laplacian variance on the cheeks and
forehead). They are kept separate from the geometric ratios so the regions can
be verified visually in the debug overlays before any feature is computed.

All index groups assume the 478-point refined mesh (iris landmarks 468-477
require the refined model, which face_landmarker.task provides).
"""
import cv2
import numpy as np

# Outer lip contour, ordered as a closed loop (whole mouth incl. both lips).
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
              291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

# Iris rings (4 points around each iris; centre is 468 / 473).
LEFT_IRIS  = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

# Cheek skin clusters, convex-hulled so landmark order doesn't matter. Grown
# UP onto the zygomatic/infraorbital cheekbone (clean skin) rather than down
# toward the beard, nasolabial fold and jaw. Lateral edge at 116/123 (left),
# lower bound at 205/206/36/101, top bound under the eye at 117/118/119/50.
# RIGHT_* mirror LEFT_*.
LEFT_CHEEK  = [116, 123, 117, 118, 119, 50, 101, 36, 205, 206]
RIGHT_CHEEK = [345, 352, 346, 347, 348, 280, 330, 266, 425, 426]


def _poly(landmarks, idxs, hull=False):
    """Build an Nx2 int pixel polygon from landmark indices."""
    pts = np.array([[landmarks[i][0], landmarks[i][1]] for i in idxs],
                   dtype=np.int32)
    if hull:
        pts = cv2.convexHull(pts).reshape(-1, 2)
    return pts


def extract_regions(landmarks):
    """
    Return {region_name: Nx2 int polygon} for every appearance region.

    landmarks must be in pixel coordinates (same scaling as the overlay image).
    Lips use the ordered outer loop; the others are convex hulls of a small
    landmark cluster.
    """
    return {
        "lips":        _poly(landmarks, LIPS_OUTER),
        "left_iris":   _poly(landmarks, LEFT_IRIS, hull=True),
        "right_iris":  _poly(landmarks, RIGHT_IRIS, hull=True),
        "left_cheek":  _poly(landmarks, LEFT_CHEEK, hull=True),
        "right_cheek": _poly(landmarks, RIGHT_CHEEK, hull=True),
    }
