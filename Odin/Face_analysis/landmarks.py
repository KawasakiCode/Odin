import cv2
import mediapipe as mp
import numpy as np

def calculate_landmarks_array(image_path):
    # Initialize MediaPipe Face Mesh
    # With refine_landmarks=True landmarks 
    # will be 478
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(  
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # Read image with cv2
    image = cv2.imread(image_path)
    if image is None:
        face_mesh.close()
        raise FileNotFoundError(f"Could not read image at {image_path!r}")

    # Default cv2 reads images as BGR (Blue, Green, Red) but MediaPipe
    # requires RGB so we convert
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    # Close MediaPipe
    face_mesh.close()

    if not results.multi_face_landmarks:
        raise ValueError(f"No face detected in image at {image_path!r}")

    face_landmarks = results.multi_face_landmarks[0]

    landmarks_list = [
        [landmark.x, landmark.y, landmark.z]
        for landmark in face_landmarks.landmark
    ]

    # Landmarks array is a list of lists where every landmark
    # has 3 dimmensions x, y, z. Coordinates are normalised to [0, 1]
    # (x by image width, y by image height).
    landmarks_array = np.array(landmarks_list)

    # The BGR image is returned too: the appearance features (colour /
    # texture) sample the raw pixels, and need pixel-space landmarks scaled
    # by this image's dimensions.
    return landmarks_array, image