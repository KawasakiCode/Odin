import numpy as np
from separate_landmarks import extract_face_data
from MediaPipe.grab_face import calculate_landmarks_array

IMAGEPATH = "ODIN/image.jpg"

landmarks = calculate_landmarks_array[IMAGEPATH]

def calculate_euclidian_distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)

face_data = extract_face_data(landmarks)

