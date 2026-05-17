from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.Ratios.calculate_ratios import canthal_tilt, fwhr, height_ratio_36, symmetry_score, width_ratio_46
from Odin.Face_analysis.constants import IMAGEPATH

landmarks = calculate_landmarks_array(IMAGEPATH)
face_data = extract_face_data(landmarks)

results = {
    "symmetry": symmetry_score(face_data),
    "width_ratio_46": width_ratio_46(face_data),
    "height_ratio_36": height_ratio_36(face_data),
    "canthal_tilt": canthal_tilt(face_data),
    "fwhr": fwhr(face_data),
}
