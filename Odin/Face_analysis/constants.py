# The path of the image to get rated
IMAGEPATH = "image.jpg"

# Which trained model to use for this photo: "male" or "female".
# Each sex has its own model in models/model_<sex>.joblib.
SEX = "female"

# Path to the MediaPipe Tasks face landmarker model bundle. This file is NOT
# shipped in the repo — download face_landmarker.task (the refined bundle that
# includes the 478 iris-aware landmarks) and place it at this path, e.g. from:
#   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# Resolved relative to the project root in landmarks.py.
FACE_LANDMARKER_TASK = "models/face_landmarker.task"

# A list of pairs of all the mediapipe landmarks that are used
# For example left eye outer corner is landmark[33] and right eye is landmark[263]
# Compare x coordinates of all these pairs to get a symmetry score 
SYMMETRIC_PAIR_KEYS = [
    ("left_eye_outer_corner",          "right_eye_outer_corner"),
    ("left_eye_inner_corner",          "right_eye_inner_corner"),
    ("left_upper_eyelid_center",       "right_upper_eyelid_center"),
    ("left_lower_eyelid_center",       "right_lower_eyelid_center"),
    ("left_eyebrow_upper_outer_point", "right_eyebrow_upper_outer_point"),
    ("left_eyebrow_upper_inner_point", "right_eyebrow_upper_inner_point"),
    ("left_eyebrow_lower_outer_point", "right_eyebrow_lower_outer_point"),
    ("left_eyebrow_lower_inner_point", "right_eyebrow_lower_inner_point"),
    ("left_eyebrow_peak_from_eye",     "right_eyebrow_peak_from_eye"),
    ("left_eyebrow_peak_from_forehead","right_eyebrow_peak_from_forehead"),
    ("left_alare_tip",                 "right_alare_tip"),
    ("lip_left_outer",                 "lip_right_outer"),
    ("left_zygomatic",                 "right_zygomatic"),
    ("left_jaw_angle1",                "right_jaw_angle1"),
    ("left_jaw_angle2",                "right_jaw_angle2"),
    ("left_cheek_apex",                "right_cheek_apex"),
    ("left_cheek_hollow",              "right_cheek_hollow"),
]