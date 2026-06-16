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

## Importing the model into another project
The `.joblib` model file is **not** self-contained: a prediction is only valid if
the 40 input features are computed by this exact pipeline (same landmarks, same
pixel scaling, same ratio definitions, same column order). So you must bring the
feature-extraction code along with the model, not just the bundle.

**Files to copy** (preserve the folder layout — paths are resolved relative to the
project root, i.e. the folder that contains the `Odin/` package):

```
Odin/                                   # the feature-extraction package
├── __init__.py
├── main.py                             # build_features() + male_boost() helpers
└── Face_analysis/
    ├── __init__.py
    ├── constants.py                    # feature keys + male-boost constants
    ├── landmarks.py
    ├── face_data.py
    ├── utils.py
    └── Ratios/
        ├── __init__.py
        ├── calculate_ratios.py
        ├── appearance.py
        └── regions.py
models/
├── model_male.joblib                   # XGBoost bundle (male)
├── model_female.joblib                 # XGBoost bundle (female)
└── face_landmarker.task                # MediaPipe Tasks model (download separately)
```

**Python dependencies:** `mediapipe`, `opencv-python`, `numpy`, `pandas`, `joblib`,
`xgboost`, `scikit-learn` (xgboost's sklearn wrapper needs it to load).

**Each bundle is a dict:** `xgboost` (the regressor), `feature_names` (the required
column order), `label` (`"MALE"`/`"FEMALE"`), `target`, `n_samples`, and the male
calibration stats `xgb_pred_mean` / `xgb_pred_max`.

**Minimal usage** (from the project root):
```python
import joblib, pandas as pd
from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.Ratios.appearance import appearance_features
from Odin.main import build_features, male_boost

landmarks, img = calculate_landmarks_array("photo.jpg")
h, w = img.shape[:2]
px = landmarks.copy(); px[:, 0] *= w; px[:, 1] *= h; px[:, 2] *= w  # pixel space
features = build_features(extract_face_data(px), appearance_features(img, px))

model = joblib.load("models/model_male.joblib")
X = pd.DataFrame([features], columns=model["feature_names"])
score = float(model["xgboost"].predict(X)[0])
if model["label"] == "MALE":          # male-only presentation boost
    score = male_boost(score, model)
print(round(score, 2))
```

## Model performance
Trained on **SCUT-FBP5500** (2,750 faces per sex, attractiveness rescaled to 1-10).
Metrics below are **5-fold cross-validated** (held-out, not training-fit):

| Model | MAE | MSE | RMSE | R² |
|-------|-----|-----|------|-----|
| Female | 0.774 | 1.004 | 1.002 | 0.611 |
| Male (raw XGBoost) | 0.695 | 0.873 | 0.934 | 0.591 |
| Male (with deployed boost) | 0.859 | 1.452 | 1.205 | 0.320 |

The male **boost** is a presentation-layer transform that stretches strong faces
upward toward a Western/FaceIQ-style scale. It deliberately trades fit against
SCUT's own raters (hence the higher MAE/RMSE and lower R² in the last row) for a
more intuitive score on out-of-distribution photos. The raw male row reflects the
model's true predictive accuracy on its training distribution.
