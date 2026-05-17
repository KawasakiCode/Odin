import numpy as np
import math
from Odin.Face_analysis.constants import SYMMETRIC_PAIR_KEYS

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

# Optimal Length/Width Midface Ratios
def  width_ratio_46(face_data):
    """
    The 46% ratio calculates the distance from the 
    center of the left pupil to the center of the 
    right pupil. That distance should be 46% of 
    the bizygomatic width.

    Ideal ratio -> 0.46
    < 0.40 -> close-set eye (or very wide cheekbones)
    > 0.50 -> wide-set eyes
    """
    left_pupil_x = face_data["left_pupil_center"][0]
    left_pupil_y = face_data["left_pupil_center"][1]

    right_pupil_x = face_data["right_pupil_center"][0]
    right_pupil_y = face_data["right_pupil_center"][1]

    # Interpupillary Distance
    IPD = np.linalg.norm(
        np.array([left_pupil_x, left_pupil_y]) -
        np.array([right_pupil_x, right_pupil_y])
    )

    # Bizygomatic Width
    biz_width = np.linalg.norm(  
        face_data["left_zygomatic"] - 
        face_data["right_zygomatic"]
    )

    return IPD / biz_width

def height_ratio_36(face_data):
    """
    The 36% ratio calculates the vertical distance from the 
    horizontal line of the pupils down to the 
    horizontal line of the mouth (where the lips meet).
    The distance should be 36% of the total height of the face
    (Trichion to Menton)

    Ideal ratio -> 0.36
    < 0.33 -> compressed midface (features sit too high)
    > 0.40 -> elongated midface (featues sit too low)
    """
    pupils_y = (
        face_data["left_pupil_center"][1] + 
        face_data["right_pupil_center"][1]
    ) / 2

    # Stomion: midpoint between upper and lower lip meeting point
    stomion_y = (
        face_data["upper_lip_bottom_center"][1] + 
        face_data["lower_lip_top_center"][1]
    ) / 2

    # Trichion: The center of the hairline
    trichion_y = face_data["top_center_forehead"][1]
    # Menton: Bottom of chin
    menton_y = face_data["chin"][1]

    eye_to_mouth = abs(pupils_y - stomion_y)
    face_height = abs(trichion_y - menton_y)

    return eye_to_mouth / face_height

# Canthal Tilt
def canthal_tilt(inner_canthus, outer_canthus):
    dx = outer_canthus[0] - inner_canthus[0]
    dy = outer_canthus[1] - inner_canthus[1]
    return math.degrees(math.atan2(-dy, dx))

def canthal_tilt_final(face_data):
    """
    Canthal tilt is the angle formed by the line from the
    inner corner of the eye to the outer corner.
    Positive = outer corner higher than inner.
    Negative = outer corner lower than inner.

    Ideal: Male: +3° to +5° / Female: +5° to 8°
    Neutral: 0° to +3°
    Negative: <0°  
    """

    left_tilt = canthal_tilt(
        face_data["left_eye_inner_corner"], 
        face_data["left_eye_outer_corner"])
    
    right_tilt = canthal_tilt(
        face_data["right_eye_inner_corner"], 
        face_data["right_eye_outer_corner"])
    
    avg_tilt = (left_tilt + right_tilt) / 2

    return avg_tilt

def fwhr(face_data):
    """
    Facial Width-to-Height Ratio. Divides bizygomatic width
    by upper face height (nasion to stomion).

    Ideal: Male 1.9-2.05  /  Female 1.75-1.9
    < 1.7  -> too narrow
    > 2.2  -> disproportionately wide
    """
    biz_width = np.linalg.norm(
        face_data["left_zygomatic"] - face_data["right_zygomatic"]
    )
    upper_face_height = np.linalg.norm(
        face_data["eyebrows_bottom"] - face_data["upper_lip_top_center"]
    )

    return biz_width / upper_face_height

