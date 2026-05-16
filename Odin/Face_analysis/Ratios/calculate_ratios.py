import numpy as np
from constants import SYMMETRIC_PAIR_KEYS

def calculate_euclidian_distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)

# Bilateral Symmetry
def symmetry_score(face_data):
    """
    For each left/right pair:
      - Mirror the left landmark by flipping x around midline
      - Compute Euclidean distance to the right landmark
      - 
    """
    
    midline_landmarks = [
        face_data["top_center_forehead"],
        face_data["glabella"],
        face_data["top_of_nose_bridge"],
        face_data["nose_tip"],
        face_data["base_of_nose"],
        face_data["upper_lip_top_center"],
        face_data["chin"]
    ]

    midline_x = np.mean([lm[0] for lm in midline_landmarks])
    
    diffs = []
    for left_key, right_key in SYMMETRIC_PAIR_KEYS:
        L = face_data[left_key]
        R = face_data[right_key]

        mirrored_lx = 2 * midline_x - L[0]
        mirrored_ly = L[1]

        # Distance between mirrored left and actual right
        dx = mirrored_lx - R[0]
        dy = mirrored_ly - R[1]

        diffs.append(dx)
        diffs.append(dy)

    euclidean_norm = np.linalg.norm(np.array(diffs))

    # Score: 0 = perfect symmetry (lower is better)
    return euclidean_norm


