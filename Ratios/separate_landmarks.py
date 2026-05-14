import numpy as np
from MediaPipe.grab_face import calculate_landmarks_array

IMAGEPATH = "ODIN/image.jpg"

landmarks = calculate_landmarks_array[IMAGEPATH]

# Left Eye Contour
left_upper_eyelid_center = landmarks[159]
left_lower_eyelid_center = landmarks[145]
left_eye_outer_corner = landmarks[33]
left_eye_inner_corner = landmarks[133]

# Right Eye Contour
right_upper_eyelid_center = landmarks[386]
right_lower_eyelid_center = landmarks[374]
right_eye_outer_corner = landmarks[263]
right_eye_inner_conter = landmarks[362]

# Left Eyebrow
left_eyebrow_upper_outer_point = landmarks[70]
left_eyebrow_upper_inner_point = landmarks[107]
left_eyebrow_lower_outer_point = landmarks[46]
left_eyebrow_lower_inner_point = landmarks[55]
left_eyebrow_peak_from_eye = landmarks[52]
left_eyebrow_peak_from_forehead = landmarks[105]

# Right Eyebrow
right_eyebrow_upper_outer_point = landmarks[300]
right_eyebrow_upper_inner_point = landmarks[336]
right_eyebrow_lower_outer_point = landmarks[276]
right_eyebrow_lower_inner_point = landmarks[285]
right_eyebrow_peak_from_eye = landmarks[282]
right_eyebrow_peak_from_forehead = landmarks[334]

# Nose
base_of_nose = landmarks[2]
top_of_nose_bridge = landmarks[168]
nose_tip = landmarks[4]
left_alare_tip = landmarks[129]
right_alare_tip = landmarks[358]

# Lips
lip_left_outer = landmarks[61]
lip_right_outer = landmarks[291]
upper_lip_top_center = landmarks[0]
upper_lip_bottom_center = landmarks[13]
lower_lip_top_center = landmarks[14]
lower_lip_bottom_center = landmarks[17]

# Jawline and Oval
chin = landmarks[152]
left_zygomatic = landmarks[234]
right_zygomatic = landmarks[454]
left_jaw_angle1 = landmarks[132]
left_jaw_angle2 = landmarks[58]
right_jaw_angle1 = landmarks[361]
right_jaw_angle2 = landmarks[288]

# Forehead
top_center_forehead = landmarks[10]
glabella = landmarks[9]

# Cheeks
left_cheek_apex = landmarks[50]
right_cheek_apex = landmarks[280]