from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.Ratios.calculate_ratios import symmetry_score
from Odin.Face_analysis.constants import IMAGEPATH

landmarks = calculate_landmarks_array(IMAGEPATH)
face_data = extract_face_data(landmarks)

results = {
    "symmetry": symmetry_score(face_data)
}
