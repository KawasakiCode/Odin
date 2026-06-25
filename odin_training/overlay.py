"""
Debug overlay helper for the SCUT extraction pipeline: draws the appearance-
sampling regions (lips / irises / cheeks / forehead) on an image so their
placement can be verified before any colour/texture feature is computed.
"""
from pathlib import Path

import cv2

from Odin.Face_analysis.Ratios.regions import extract_regions

# Anchored to this file's folder so it works from any working directory.
DEBUG_OVERLAY_DIR = Path(__file__).resolve().parent / "debug_overlays"

# BGR colour per appearance region, drawn as a translucent fill + solid outline.
REGION_COLORS = {
    "lips":        (0, 0, 255),     # red
    "left_iris":   (0, 255, 0),     # green
    "right_iris":  (0, 255, 0),
    "left_cheek":  (255, 0, 0),     # blue
    "right_cheek": (255, 0, 0),
    "forehead":    (0, 255, 255),   # yellow
}


def save_landmark_overlay(mp_image, landmarks_array, face_data, out_path):
    """
    Save a copy of the image with the appearance-sampling regions drawn on top.

    Each region is a translucent coloured fill with a solid outline; no landmarks
    or measurement guides are drawn. landmarks_array must be in pixel coordinates
    (as scaled in the extraction pipeline), matching the image drawn on here.
    """
    # numpy_view() is RGB; OpenCV draws and writes in BGR
    img = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR).copy()

    regions = extract_regions(landmarks_array)

    # Translucent fills first (blended onto a copy), then crisp outlines on top.
    overlay = img.copy()
    for name, poly in regions.items():
        cv2.fillPoly(overlay, [poly], REGION_COLORS[name])
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    for name, poly in regions.items():
        cv2.polylines(img, [poly], isClosed=True, color=REGION_COLORS[name],
                      thickness=1, lineType=cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
