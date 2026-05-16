# A list of pairs of all the mediapipe landmarks that are used
# For example left eye outer corner is landmark[33] and right eye is landmark[263]
# Compare x coordinates of all these pairs to get a symmetry score 
SYMMETRIC_PAIR_KEYS = [
    ("left_eye_outer_corner",          "right_eye_outer_corner"),
    ("left_eye_inner_corner",          "right_eye_inner_conter"),
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