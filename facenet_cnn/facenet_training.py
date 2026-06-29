from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV = str(BASE.parent / "odin_training" / "training_data_scut.csv")

df = pd.read_csv(CSV)
print(f"Loaded {len(df)} faces, {df.shape[1]} columns from cache")

npz = np.load(BASE / "embeddings_scut.npz", allow_pickle=True)

X_male, X_female, y_male, y_female = [], [], [], []
scores = dict(zip(df["Image_ID"], df["Attractiveness"]))

for id, embeddings in zip(npz["ids"], npz["landmarks"]):
    if id[1] == "F":
        X_female.append(embeddings)
        y_female.append(scores[id])
    else: 
        X_male.append(embeddings)
        y_male.append(scores[id])