import numpy as np
import math
from Odin.Face_analysis.constants import SYMMETRIC_PAIR_KEYS
from Face_analysis.utils import angle_at_vertex, line_angle_degrees

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

# Facial width to height ratio
def fwhr(face_data):
    """
    Facial Width-to-Height Ratio. Divides bizygomatic width
    by upper face height (nasion to stomion).

    The stomion is approximated as the midpoint between
    upper_lip_bottom_center and lower_lip_top_center.

    Ideal: Male 1.9-2.05  /  Female 1.75-1.9
    < 1.7  -> too narrow
    > 2.2  -> disproportionately wide
    """
    biz_width = np.linalg.norm(
        face_data["left_zygomatic"] - face_data["right_zygomatic"]
    )
    bottom_eyebrows = face_data["eyebrows_bottom"]
    upper_lip_top = face_data["upper_lip_top_center"]

    upper_face_height = np.linalg.norm(bottom_eyebrows - upper_lip_top)

    return biz_width / upper_face_height

# Frontal jaw contour angle
# TODO probably add this ratio to the side profile picture as gonial angle
def frontal_jaw_contour_angle(face_data):
    """
    The frontal jaw contour angle measures the slope of the jawline
    in a frontal photo, compared against a reference line from the
    lateral canthus to the ipsilateral alare (nostril tip).
    
    Unlike the gonial angle (measured from a profile photo), this
    angle is wider and measures how parallel the jawline runs
    relative to the canthus-alare reference line.

    A deviation of 0° means the jawline is perfectly parallel to
    the canthus-alare line. The jawline should slope downward
    slightly from the jaw angle toward the chin.

    Ideal: 0°-15° downward deviation from the canthus-alare line
    Female frontal jaw angle: ~139°-142° (wider, more tapered)
    Male frontal jaw angle:   ~125°-130° (squarer, more defined)
    > 15° deviation -> jawline drops too steeply (weak jaw)
    = 0°            -> perfectly parallel (very defined jaw)
    """

    left_ref_angle = line_angle_degrees(  
        face_data["left_eye_outer_corner"],
        face_data["left_alare_tip"]
    )
    left_jaw_angle = line_angle_degrees(  
        face_data["left_jaw_angle1"],
        face_data["chin"]
    )
    left_deviation = abs(left_ref_angle - left_jaw_angle)


    right_ref_angle = line_angle_degrees(  
        face_data["right_eye_outer_corner"],
        face_data["right_alare_tip"]
    )
    right_jaw_angle = line_angle_degrees(  
        face_data["right_jaw_angle1"],
        face_data["chin"]
    )
    right_deviation = abs(right_ref_angle - right_jaw_angle)

    return {
        "deviation": (left_deviation + right_deviation) / 2,
        "jaw_slope": (left_jaw_angle + right_jaw_angle) / 2,
        "canthus_alare_slope": (left_ref_angle + right_ref_angle) / 2
    }

# Facial Thirds
def horizontal_thirds(face_data):
    """
    Divides the face into three vertical regions and measures
    each as a percentage of total face height.

    Upper third:  trichion (hairline) -> glabella (between brows)
    Middle third: glabella -> subnasale (base of nose)
    Lower third:  subnasale -> menton (chin)

    Classical ideal: 1:1:1 (33.3% each)

    Modern gender-adjusted ideals:
    Male:   upper 31.0% / middle 30.5% / lower 38.5%
    Female: upper 29.5% / middle 32.4% / lower 38.2%

    Both sexes show a dominant lower third — the classical
    1:1:1 canon is outdated for attractiveness assessment.
    """   
    upper = abs(face_data["top_center_forehead"][1] - face_data["glabella"][1])
    middle = abs(face_data["glabella"][1] - face_data["base_of_nose"][1])
    lower = abs(face_data["base_of_nose"][1] - face_data["chin"][1])

    total = upper + middle + lower

    upper_percentage = upper / total
    middle_percentage = middle / total
    lower_percentage = lower / total

    return {
        "upper_perc": upper_percentage,
        "middle_perc": middle_percentage,
        "lower_perc": lower_percentage
    }

# Bizygomatic / Bigonial Ratio
def bizygomatic_bigonial_ratio(face_data):
    """
    Divides bizygomatic width (cheekbone to cheekbone) by
    bigonial width (jaw angle to jaw angle). Measures how
    much the cheekbones flare relative to the jaw — the
    degree of facial taper from midface to lower face.

    Ideal: Female 1.174 (more tapered, V-shaped)
           Male   1.128 (squarer, less taper)
    < 1.0  -> jaw wider than cheekbones (very masculine/unusual)
    > 1.3  -> extremely tapered (very feminine/heart-shaped)
    """    
    biz_width = np.linalg.norm(  
        face_data["left_zygomatic"] - face_data["right_zygomatic"]
    )

    big_width = np.linalg.norm(
        face_data["left_jaw_angle1"] - face_data["right_jaw_angle1"]
    )

    return biz_width / big_width

# Facial Fifths
def facial_fifths(face_data):
    """
    Divides the total face width into fifths, each ideally
    equal to one eye width. Assesses horizontal facial harmony.

    Measures:
    - face_width / avg_eye_width        -> ideal 4.0 - 4.25
    - intercanthal_dist / avg_eye_width -> ideal 0.95 - 1.15
      (gap between eyes should equal one eye width)

    > 4.5 face ratio  -> face too wide for eye spacing
    < 3.8 face ratio  -> face too narrow for eye spacing
    > 1.15 inter-eye   -> eyes too far apart
    < 0.95 inter-eye   -> eyes too close together
    """

    left_eye_width = np.linalg.norm(
        face_data["left_eye_outer_corner"] - face_data["left_eye_inner_corner"]
    )    

    right_eye_width = np.linalg.norm(
        face_data["right_eye_outer_corner"] - face_data["right_eye_inner_corner"]
    )   

    avg_eye_width = (left_eye_width + right_eye_width) / 2

    face_width = np.linalg.norm(  
        face_data["left_zygomatic"] - face_data["right_zygomatic"]
    )

    intercanthal = np.linalg.norm(
        face_data["left_eye_inner_corner"] - face_data["right_eye_inner_corner"]
    )

    return {
        "fifths_ratio": face_width / avg_eye_width,
        "inter_eye_ratio": intercanthal / avg_eye_width
    }

# Orbitonasal / Nose-Intercanthal
def orbitonasal_ratio(face_data):
    """
    Compares nose width (ala to ala) against the intercanthal
    distance (inner corner to inner corner of the eyes).
    One of the original Neoclassical Canons.

    Ideal: 1.0 (nose width equals intercanthal distance)
    Female exception: attractive female faces often score
    slightly below 1.0, signaling a smaller, more delicate nose.

    > 1.2 -> nose too wide relative to eye spacing
    < 0.8 -> nose too narrow (or eyes too close together)
    """  
    nose_width = np.linalg.norm(  
        face_data["left_alare_tip"] - face_data["right_alare_tip"]
    )
    intercanthal = np.linalg.norm(
        face_data["left_eye_inner_corner"] - face_data["right_eye_inner_corner"]
    )

    return nose_width / intercanthal 

# Nasofacial Proportion
def nasofacial_proportion(face_data):
    """
    Compares nose width against total bizygomatic face width.
    The nose should occupy exactly one quarter of the face width.

    Ideal: 0.25 (nose width = 25% of face width)
    > 0.30 -> nose too wide for the face
    < 0.20 -> nose too narrow for the face
    """    
    nose_width = np.linalg.norm(
        face_data["left_alare_tip"] - face_data["right_alare_tip"]
    )
    face_width = np.linalg.norm(  
        face_data["left_zygomatic"] - face_data["right_zygomatic"]
    )

    return nose_width / face_width

# Naso-Oral / Mouth-Nose Ratio
def naso_oral_ratio(face_data):
    """
    Compares mouth width (cheilion to cheilion) against
    nose width (ala to ala). One of the Neoclassical Canons.
    The mouth should be 1.5 times the width of the nose.

    Ideal: 1.5 - 1.62 (mouth width = 150% of nose width)
    > 1.7 -> mouth too wide relative to nose
    < 1.3 -> mouth too narrow relative to nose
    """    
    mouth_width = np.linalg.norm(
        face_data["lip_left_outer"] - face_data["lip_right_outer"]
    )
    nose_width  = np.linalg.norm(
        face_data["left_alare_tip"] - face_data["right_alare_tip"]
    )

    return mouth_width / nose_width

# Face Golden Ratio
def face_golden_ratio(face_data):
    """
    Divides total face height (trichion to menton) by
    bizygomatic width. Evaluates the overall face bounding
    box proportions against the Golden Ratio.

    Ideal: Female: 1.30 / Male: 1.35

    < 1.3  -> face too wide (very broad/flat appearance)
    > 1.4  -> face too elongated (very narrow appearance)
    """    
    face_height = np.linalg.norm(
        face_data["top_center_forehead"] - face_data["chin"]
    )
    face_width  = np.linalg.norm(
        face_data["left_zygomatic"] - face_data["right_zygomatic"]
    )

    return face_height / face_width

# Face Height/Bigonial Width
def face_height_bigonial_width(face_data):
    """
    Divides total face height (trichion to menton) by
    bigonial width (jaw angle to jaw angle).

    Ideal: Female 1.613 (±0.063) — virtually exact Golden Ratio
           Male   1.566 (±0.085) — slightly wider jaw
    < 1.4  -> jaw too wide for face height (very square)
    > 1.7  -> jaw too narrow for face height (very tapered)
    """    
    face_height = np.linalg.norm(
        face_data["top_center_forehead"] - face_data["chin"]
    )
    big_width   = np.linalg.norm(
        face_data["left_jaw_angle1"] - face_data["right_jaw_angle1"]
    )

    return face_height / big_width