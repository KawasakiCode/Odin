import cv2
import mediapipe as mp
import numpy as np

def calculate_landmarks_array(image_path):
    # Initialize MediaPipe Face Mesh
    # With refine_landmarks=False landmarks 
    # will be 468 instead of 478
    # The extra 10 are used to track the iris of the eyes which are irrelevant
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(  
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    # Read image with cv2
    image = cv2.imread(image_path)

    # Default cv2 reads images as BGR (Blue, Green, Red) but MediaPipe
    # requires RGB so we convert
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BRG2RGB)

    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        landmarks_list = [
            [landmark.x, landmark.y, landmark.z]
            for landmark in face_landmarks.landmark
        ]

        # Landmarks array is a list of lists where every landmark
        # has 3 dimmensions x, y, z
        landmarks_array = np.array(landmarks_list)

    # Close MediaPipe
    face_mesh.close()

    return landmarks_array