# Facial Attractiveness Ratio Analyzer
This project analyzes a photograph of a human face to calculate structural proportions and assess attractiveness based on ratios established in scientific literature. It uses Google's MediaPipe AI to extract a highly detailed 3D topographical map of the face and NumPy to calculate the exact distances and ratios between specific facial landmarks.
## Features
 * **3D Facial Landmarking:** Utilizes MediaPipe Face Mesh to extract 478 precise (X, Y, Z) coordinates from a single 2D image.
 * **Accurate Distance Calculation:** Uses 3D Euclidean distance math via NumPy to ensure that slight head tilts or angles do not distort the morphological measurements.
 * **Scientific Ratio Analysis:** Computes facial ratios (e.g., facial thirds, golden ratio, eye-to-mouth distances) to compare against established scientific literature on facial aesthetics.
## Requirements
The project requires Python 3 and the following libraries:
 * mediapipe (for facial landmark extraction)
 * opencv-python (for image loading and processing)
 * numpy (for high-performance 3D vector math and distance calculations)
## Installation
Install the required dependencies using pip:
```bash
pip install mediapipe opencv-python numpy

```
## Usage
 1. Place the target facial photograph (e.g., photo.jpg) in the project directory.
 2. Update the image_path variable in the script to match your file's name.
 3. Run the script.
The script will:
 1. Load the image and convert it to the RGB color space.
 2. Pass the image through the MediaPipe Face Mesh model.
 3. Extract the 478 facial landmarks into a structured (478, 3) NumPy array.
 4. Calculate 3D Euclidean distances between specific topology indices to compute the final ratios.
## Finding Landmark Indices
MediaPipe does not assign string names to the landmarks. To calculate specific ratios, you must reference an official **MediaPipe Face Mesh Topology Map** to find the exact index numbers (0-477) for the required facial points (e.g., the tip of the nose is index 1).
