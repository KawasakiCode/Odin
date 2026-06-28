"""
One-time extraction of raw landmarks for every SCUT face, cached to
landmarks_scut.npz so the Procrustes/shape features can be computed without
re-running MediaPipe each time.

Saves pixel-space (x, y) for all 478 mesh points per face (SCUT images are
square, so pixel == aspect-true). Keyed by Image_ID to merge with the CSV.
"""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from tqdm import tqdm

# Anchor paths to this file's folder so it works from any working directory.
BASE = Path(__file__).resolve().parent
IMAGES_DIR = BASE / "scut" / "Images"
OUT = str(BASE / "landmarks_scut.npz")
TASK_MODEL = str(BASE.parent / "models" / "face_landmarker.task")


def main():
    paths = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Extracting landmarks for {len(paths)} images...")

    base = python.BaseOptions(model_asset_path=TASK_MODEL)
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(opts)

    ids, arrs = [], []
    misses = 0
    for p in tqdm(paths):
        try:
            img = mp.Image.create_from_file(str(p))
        except Exception:
            misses += 1
            continue
        res = detector.detect(img)
        if not res.face_landmarks:
            misses += 1
            continue
        w, h = img.width, img.height
        lm = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]],
                      dtype=np.float32)  # (478, 2)
        ids.append(p.name)
        arrs.append(lm)

    detector.close()
    data = np.stack(arrs)  # (N, 478, 2)
    np.savez_compressed(OUT, ids=np.array(ids), landmarks=data)
    print(f"Saved {OUT}: {data.shape} ({misses} faces skipped)")


if __name__ == "__main__":
    main()
