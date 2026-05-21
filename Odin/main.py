from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.Ratios.calculate_ratios import bizygomatic_bigonial_ratio, canthal_tilt, face_golden_ratio, face_height_bigonial_width, facial_fifths, frontal_jaw_contour_angle, fwhr, height_ratio_36, horizontal_thirds, naso_oral_ratio, nasofacial_proportion, orbitonasal_ratio, symmetry_score, width_ratio_46
from Odin.Face_analysis.constants import IMAGEPATH

landmarks = calculate_landmarks_array(IMAGEPATH)
face_data = extract_face_data(landmarks)

results = {
    "symmetry": symmetry_score(face_data),
    "width_ratio_46": width_ratio_46(face_data),
    "height_ratio_36": height_ratio_36(face_data),
    "canthal_tilt": canthal_tilt(face_data),
    "fwhr": fwhr(face_data),
    "frontal_jaw_contour_angle": frontal_jaw_contour_angle(face_data),
    "horizontal_thirds": horizontal_thirds(face_data),
    "bizygomatic_bigonial_ratio": bizygomatic_bigonial_ratio(face_data),
    "facial_fifths": facial_fifths(face_data),
    "orbitonasal_ratio": orbitonasal_ratio(face_data),
    "nasofacial_proportion": nasofacial_proportion(face_data),
    "naso_oral_ratio": naso_oral_ratio(face_data),
    "face_golden_ratio": face_golden_ratio(face_data),
    "face_height_bigonial_width": face_height_bigonial_width(face_data),
    
}
