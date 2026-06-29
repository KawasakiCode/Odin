from pathlib import Path
from tqdm import tqdm
import numpy as np
from deepface import DeepFace

BASE = Path(__file__).resolve().parent
SCUT_IMAGES = BASE.parent / "odin_training" / "scut" / "Images"
OUT = BASE / "embeddings_scut.npz"

paths = sorted(SCUT_IMAGES.glob("*.jpg"))

embeddings, ids = [], []
misses = 0

for p in tqdm(paths): # Run Facenet on all SCUT Images
    try:
        reps = DeepFace.represent(
            img_path=str(p),
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False,
        )
        embedding = reps[0]["embedding"]
        embeddings.append(embedding)
        ids.append(p.name)
    except Exception as e:
        misses += 1
        print(p, e)

data = np.stack(embeddings) # (N, 512)
np.savez_compressed(OUT, ids=np.array(ids), embeddings=data)
print(f"Saved {OUT}: {data.shape} ({misses} faces skipped)")