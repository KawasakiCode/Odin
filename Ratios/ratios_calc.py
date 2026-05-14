import numpy as np
from MediaPipe.grab_face import calculate_landmarks_array

IMAGEPATH = "ODIN/image.jpg"

landmarks = calculate_landmarks_array[IMAGEPATH]
