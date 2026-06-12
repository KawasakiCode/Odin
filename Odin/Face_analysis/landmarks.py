import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from Odin.Face_analysis.constants import FACE_LANDMARKER_TASK

# Project root = two levels up from this file (Odin/Face_analysis/landmarks.py),
# used to resolve the model bundle path regardless of the working directory.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _resolve_model_path():
    """Absolute path to the face_landmarker.task bundle, with a clear error."""
    path = FACE_LANDMARKER_TASK
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MediaPipe face landmarker model not found at {path!r}. "
            "Download face_landmarker.task and place it there (see "
            "FACE_LANDMARKER_TASK in constants.py)."
        )
    return path


def calculate_landmarks_array(image_path):
    """
    Run the MediaPipe Tasks FaceLandmarker on the image and return
    (landmarks_array, bgr_image).

    The legacy mp.solutions.face_mesh API is absent from recent MediaPipe
    Windows wheels, so we use the Tasks FaceLandmarker, which detects the same
    478 refined landmarks (iris points 468-477 included). Coordinates are
    normalised to [0, 1] (x by image width, y by image height).
    """
    model_path = _resolve_model_path()

    # Resolve the image path against the project root when relative, so it is
    # found regardless of the working directory (IDE run button vs `python -m`).
    # cv2.imread silently resolves a relative path against the CWD and returns
    # None if it isn't there — which looks like "no image" even when the file
    # exists elsewhere.
    resolved_path = image_path
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.join(_PROJECT_ROOT, resolved_path)

    # Read image with cv2 (BGR). The BGR image is returned too: the appearance
    # features (colour / texture) sample raw pixels and need pixel-space
    # landmarks scaled by this image's dimensions.
    image = cv2.imread(resolved_path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image at {resolved_path!r} (IMAGEPATH={image_path!r}). "
            "Make sure the file exists there and is a readable image (jpg/png/etc)."
        )

    # MediaPipe expects RGB; cv2 gives BGR.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError(f"No face detected in image at {resolved_path!r}")

    face_landmarks = result.face_landmarks[0]

    # Each landmark has x, y, z (z roughly normalised by image width).
    landmarks_array = np.array(
        [[lm.x, lm.y, lm.z] for lm in face_landmarks]
    )

    return landmarks_array, image
