from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, KFold

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

def Ridge(X, y, sex):
    model = make_pipeline(  
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 13)) # tries alpha = 0.001 ... 1000
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    res = cross_validate(model, X, y, cv=kf, scoring=scoring)

    print(sex)
    print(f"R2  : {res['test_r2'].mean():.3f} ± {res['test_r2'].std():.3f}")
    print(f"MAE : {-res['test_neg_mean_absolute_error'].mean():.3f}")
    print(f"RMSE: {-res['test_neg_root_mean_squared_error'].mean():.3f}")

Ridge(X_male, y_male, "Male")
Ridge(X_female, y_female, "Female")