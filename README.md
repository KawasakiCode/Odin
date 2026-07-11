# Odin — Facial Geometry & Attractiveness-Modeling Study

A personal project exploring **facial geometry** — the proportions and ratios cited
in facial-aesthetics literature — and, as a modeling exercise, **how well (and how
poorly) those features predict the subjective attractiveness ratings of a specific
research dataset**. It extracts 478 facial landmarks with MediaPipe, computes ~25
classical ratios plus colour/texture and Procrustes shape features, and trains
gradient-boosted models on the SCUT-FBP5500 dataset.

> ⚠️ **This is not a tool for measuring human attractiveness.**
> The output is **NOT** a meaningful, objective, or authoritative judgment of
> anyone's appearance, and should NOT be treated as one. The "score" only
> reflects the **averaged opinions of the ~60 raters** who labelled the
> SCUT-FBP5500 dataset (predominantly East-Asian faces) — it is one narrow,
> culturally specific snapshot of taste, not a standard. I built this out of
> genuine interest in facial geometry and to map, for myself, *what actually
> drives a "beauty" rating and what doesn't*. **Please don't base your
> self-image on anything here.**

## What this project is really about

The interesting question was never "what's the score" — it was: *can hand-crafted
geometric ratios explain attractiveness ratings at all, and where do they hit a
wall?* Treating it as an honest ML investigation produced clearer findings than the
predictor itself:

- **Geometry explains ~63% of the rating variance — and the *features*, not the
  model, set that ceiling.** The deployed, fully-interpretable model reaches
  cross-validated R² **0.633 (male) / 0.636 (female)**; the rest is skin/texture/
  gestalt landmark geometry cannot see, and bigger models/tuning never move it.
  Supervised PLS shape *could* push it to **0.657 / 0.666**, but only via unnamed
  black-box axes — a trade I declined to keep every feature explainable (see the
  shape bullet).
- **"Attractiveness" doesn't transfer across rater populations for male faces.**
  A model trained on SCUT ranks held-out SCUT males well (Spearman ≈ 0.76) but
  ranks a different population's male faces (CFD) at **≈ 0.04 — essentially
  random.** For female faces it transfers moderately (≈ 0.35). In other words,
  *male* "attractiveness" as labelled here is highly population-specific.
- **Model choice matters in a teachable way.** A RandomForest, being a
  leaf-averaging model, regresses the tails toward the crowd mean (it literally
  can't rate an 8/10 face as an 8) — so it was dropped in favour of **XGBoost**,
  whose boosting follows the full curve.
- **Supervised shape (PLS) is the biggest lever — but its most useful axes resist
  naming.** Swapping unsupervised shape-PCA for **PLS** (each axis oriented toward
  the *rating*, not raw variance) roughly doubles what geometry contributes:
  averageness + 25 PLS axes add **+0.044 (female) / +0.065 (male) R²** over the
  ratios (10-seed OOF), reaching 0.666 / 0.657. But leave-one-out and forward
  add-back ablations showed that lift is **diffuse across many correlated axes**,
  and the axes carrying it don't map to nameable deformations. So the deployed
  model keeps only the **named shape axes** (~7 per sex, each a labelled
  deformation mode) and sits at **0.636 / 0.633** — a deliberate ~0.03 R² cost for
  a model where every feature is explainable. For males, one named axis (jaw
  angularity × face width-to-height) is still the single top feature.
- **Hand-crafted skin descriptors barely move the needle.** To test whether the
  "skin quality" part of the ceiling gap is recoverable *by design*, I added
  colour-unevenness (CIELab a\*/b\* spread → pigmentation/blotchiness) and a
  band-pass blemish detector. Under the same 10-seed paired ablation they add only
  **+0.006 (female) / +0.005 (male) R²** — a real lift (the sign holds), but
  negligible, and for males the seed-to-seed spread is about as large as the effect
  itself. The takeaway is the interesting part: the gap to a deep embedding is
  **not** mostly nameable skin texture waiting to be measured — it's un-interpretable
  gestalt.
- **Interpretable geometry lands ~0.12 R² below a deep embedding — by choice.** A
  frozen FaceNet-512 embedding + Ridge reaches ~0.75 R² and the deep-CNN benchmark
  ~0.81; the deployed **fully-interpretable** model reaches **~0.63** (full PLS,
  with black-box axes, would reach ~0.66 — ~0.09 below the embedding). What remains
  is un-interpretable gestalt and photographic confounds — the model trades it for
  explanations you can actually read.

I think the limitations are the most valuable part — they're documented here on
purpose.

## Results

All figures are aggregate statistics (no face images), reproducible with
`python plots/generate_plots.py`.

**Where interpretable geometry sits vs. the ceiling.** The deployed interpretable
model reaches ~0.63 R² (full PLS ~0.66); a frozen FaceNet-512 embedding ~0.75; the
deep-CNN benchmark ~0.81. I measured my own ceiling instead of guessing it.

![Ceiling comparison](plots/1_ceiling_comparison.png)

**How the model behaves — honest out-of-fold predictions.** Note the *tail
compression*: extreme faces are pulled toward the mean, because hand-crafted
features can't confidently place a 2 or a 9. This is worse in RandomForest (why it
was dropped).

![Predicted vs actual](plots/2_predicted_vs_actual.png)

**What the model leans on.** Top-15 features per sex, coloured by group — geometry
dominates, with Procrustes shape and colour/texture contributing.

![Feature importance](plots/3_feature_importance.png)

**The bias finding: male "attractiveness" doesn't transfer across rater
populations.** Trained on SCUT, the model ranks held-out SCUT males at Spearman
≈0.76 but a different population's males (CFD) at ≈0.04 — essentially random.
Females transfer moderately (≈0.35).

![Cross-population transfer](plots/4_cross_population_transfer.png)

**Supervised shape (PLS) is the biggest lever, verified across seeds.** Procrustes
averageness + shape-PLS add **+0.044 (female) / +0.065 (male) R²** over the ratios
(10-seed OOF), of which ~+0.03 is PLS beating PCA at the same axis count — the sign
never flips.

![Procrustes ΔR²](plots/5_procrustes_delta.png)

**The data.** SCUT-FBP5500 label distribution, 2,750 faces per sex, averaged over
~60 predominantly East-Asian raters.

![Label distribution](plots/6_label_distribution.png)

## How it works

```
photo ──► MediaPipe FaceLandmarker (478 landmarks)
      ──► geometric ratios   (thirds, fifths, golden, FWHR, jaw, EAR, …)
      ──► appearance         (skin texture + colour-unevenness + blemish, lip/eye/skin colour, CIELab contrasts)
      ──► Procrustes shape   (averageness + named PLS shape axes, ~7/sex, oriented toward the rating)
      ──► XGBoost regressor  ──► 1–10 score (+ optional male presentation boost)
```

Geometric ratios are scale-invariant; the colour/texture and shape features use
pixel-space landmarks. Models are trained per sex and bundled with their feature
order and shape basis so inference reproduces the exact feature vector.

## Project structure

```
Odin/                      # inference pipeline: image → features → score
├── main.py                #   orchestration, build_features(), male_boost()
└── Face_analysis/
    ├── landmarks.py       #   MediaPipe Tasks landmark extraction
    ├── face_data.py       #   478 raw points → named semantic points
    ├── constants.py       #   config, feature/boost constants
    ├── utils.py           #   angle helpers
    └── Ratios/
        ├── calculate_ratios.py   # the classical geometric ratios
        ├── appearance.py         # colour / texture / CIELab contrast
        ├── regions.py            # sampling polygons (lips/eyes/cheeks/forehead)
        └── shape.py              # Procrustes alignment + shape features
odin_ui/                   # web UI (upload a photo → landmarks overlay + ratios + score)
├── backend/  app.py       #   FastAPI wrapper around the exact Odin pipeline
└── frontend/              #   Vite + React + TypeScript
odin_model/                # training side (datasets NOT included — see below)
├── data_scut.py           #   SCUT feature extraction → CSV cache
├── extract_landmarks.py   #   one-time raw-landmark cache (for shape features)
├── shape_utils.py         #   GPA + shape-PLS fitting
├── train.py               #   XGBoost training, CV, bundle export
└── overlay.py             #   debug region overlays
models/                    # trained artifacts
├── model_male.joblib      #   XGBoost bundle (male)
├── model_female.joblib    #   XGBoost bundle (female)
└── face_landmarker.task   #   MediaPipe model (download separately, see below)
```

## Requirements

- **Python 3.10+**
- Inference: `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`,
  `xgboost`, `scipy`, `joblib`
- Web UI backend: `fastapi`, `uvicorn`, `python-multipart`
- Web UI frontend: **Node.js + npm**
- Training (`odin_model`): the above + `matplotlib`, `seaborn`, `tqdm`, `openpyxl`

## Installation & usage

```bash
pip install -r requirements.txt
```

Download the MediaPipe face landmarker model into `models/`:
<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>

**CLI** — set `IMAGEPATH` and `SEX` in `Odin/Face_analysis/constants.py`, then:
```bash
python -m Odin.main
# → Attractiveness (MALE): 7.64 / 10
```

**Web UI** — run the two pieces:
```bash
# backend (uses the same env as the CLI)
cd odin_ui/backend && pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# frontend (new terminal)
cd odin_ui/frontend && npm install && npm run dev   # open http://localhost:5173
```
Upload a face photo → it returns the photo with the landmark overlay, the
calculated ratios (hover a ratio to highlight the landmarks it uses, with the
male/female ideal beside it), and the model score.

**Training** (optional, requires the SCUT dataset placed under `odin_model/scut/`):
```bash
cd odin_model
python extract_landmarks.py   # one-time landmark cache
python train.py               # → 51-feature bundles in models/
```

## Data & ethics

The SCUT-FBP5500 dataset (and CFD, used briefly during the population-transfer
analysis) are **not included** in this repository — they are photographs of real
people under their own academic-use terms, and redistributing them would be a
licensing and privacy violation. Only code is published; obtain the datasets from
their original sources if you want to retrain. The ratings are subjective human
judgments from a specific, non-representative rater pool — see the disclaimer.

## Dataset citations

These datasets require citation. If you retrain or build on this work, cite their
original authors.

**SCUT-FBP5500** — primary training data:

> Liang, L., Lin, L., Jin, L., Xie, D., & Li, M. (2018). SCUT-FBP5500: A Diverse
> Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction. *2018 24th
> International Conference on Pattern Recognition (ICPR)*, 1598–1603.
> https://doi.org/10.1109/ICPR.2018.8546038

```bibtex
@inproceedings{liang2018scutfbp5500,
  title        = {{SCUT-FBP5500}: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction},
  author       = {Liang, Lingyu and Lin, Luojun and Jin, Lianwen and Xie, Duorui and Li, Mengru},
  booktitle    = {2018 24th International Conference on Pattern Recognition (ICPR)},
  pages        = {1598--1603},
  year         = {2018},
  organization = {IEEE},
  doi          = {10.1109/ICPR.2018.8546038}
}
```

**Chicago Face Database (CFD)** — used only for the cross-population transfer
analysis (not shipped, not part of the trained models). Per the CFD usage terms,
cite the set(s) used:

> CFD: Ma, D. S., Correll, J., & Wittenbrink, B. (2015). The Chicago Face Database:
> A Free Stimulus Set of Faces and Norming Data. *Behavior Research Methods, 47*,
> 1122–1135. https://doi.org/10.3758/s13428-014-0532-5
>
> CFD-INDIA: Lakshmi, A., Wittenbrink, B., Correll, J., & Ma, D. S. (2021). The
> India Face Set: International and Cultural Boundaries Impact Face Impressions and
> Perceptions of Category Membership. *Frontiers in Psychology, 12*, 627678.
> https://doi.org/10.3389/fpsyg.2021.627678
>
> CFD-MR (if you use the multiracial expansion): Ma, D. S., Kantner, J., &
> Wittenbrink, B. (2021). Chicago Face Database: Multiracial Expansion. *Behavior
> Research Methods, 53*, 1289–1300. https://doi.org/10.3758/s13428-020-01482-5

## Model performance

Trained on **SCUT-FBP5500** (2,750 faces per sex), **51 features**, scored with
**5-fold cross-validation** (held-out, not training-fit). The shape model (PLS) is
refit on each fold's train split, so the numbers are leakage-free:

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Female | 0.742 | 0.968 | 0.636 ± 0.019 |
| Male (raw XGBoost) | 0.655 | 0.884 | 0.633 ± 0.010 |

**This is the fully-interpretable model — every feature has a name.** The deployed
model keeps only the *named* shape axes (~7 per sex); the pipeline can reach
**0.666 / 0.657** by adding the ~18 unnamed "gestalt" PLS axes back, but leave-one-out
and add-back ablations showed that lift is diffuse black-box signal, so I traded
**~0.03 R²** to keep the model explainable end-to-end. That number is a deliberate
choice, not a limitation.

The labels are on a 1–10 scale (rescaled from SCUT's 1–5). For context, divide MAE
by 2.25 to compare with the deep-CNN SCUT literature (≈ 0.22 MAE on 1–5): these
hand-crafted-feature models sit at the top of the *classical/interpretable* range,
below end-to-end CNNs — expected, since they use hand-built ratios + shape instead
of raw pixels.

A male-only **presentation boost** can stretch above-mean scores toward a more
intuitive scale; it intentionally trades fit against SCUT's own raters, so it is a
display choice, not an accuracy improvement.

## How far this is from the ceiling

To see how much attractiveness signal is even *recoverable* from a face image — and
how much of it the interpretable features capture — I compared three approaches
under the **same 5-fold CV protocol** (1–10 scale):

| Approach | Male R² | Female R² | What it captures |
|----------|---------|-----------|------------------|
| Hand-crafted features — **deployed, fully interpretable** | 0.633 | 0.636 | ratios + colour + *named* PLS shape |
| Hand-crafted features — full PLS (incl. unnamed axes) | 0.657 | 0.666 | + ~18 unnamed "gestalt" shape axes |
| Frozen FaceNet-512 embedding + Ridge | 0.749 | 0.759 | signal in a face-recognition embedding |
| End-to-end CNN (SCUT-FBP5500 benchmark) | ≈ 0.81 | ≈ 0.81 | the full-image ceiling |

The CNN figure is the official **SCUT-FBP5500 benchmark** (Liang et al., 2018; see
*Dataset citations*): its best model, **ResNeXt-50**, reports a **Pearson
correlation ≈ 0.90** (so ≈ 0.81 R²) with **MAE ≈ 0.21** on the 1–5 scale under
5-fold CV. That benchmark trains on all 5,500 faces **pooled** (not per sex), so
treat ≈ 0.81 as a rough ceiling reference, not a like-for-like number.

How to read it:

- A **frozen identity embedding** — no fine-tuning at all — already recovers
  ~**0.75 R²**, about **94% of the deep-CNN ceiling**. Most of the extractable
  signal lives in a generic face representation, not in attractiveness-specific
  training.
- The **deployed interpretable model leaves ~0.12 R²** relative to that embedding.
  Full PLS shape (with the unnamed axes) narrows it to **~0.09**, but those axes are
  black-box — so the *explainable* ceiling is ~0.66 and the *deployed, fully-nameable*
  model sits ~0.03 below it by choice. What resisted hand-crafting stayed resistant:
  skin colour-unevenness + blemish descriptors closed only **~0.005**, and the
  R²-recovering shape axes turned out **diffuse and un-nameable** (leave-one-out
  said every axis was individually droppable, yet dropping them as a batch cost real
  R² — the correlated-feature trap, documented). The residual gap is holistic gestalt
  and photographic confounds that **don't reduce to nameable features**.
- **PLS shape closed the male gap.** Supervised shape helped males most (+0.065 vs
  +0.044 R²), pulling the two sexes level (the deployed interpretable models sit at
  0.633 / 0.636, near-identical) where PCA-era male geometry had trailed. That fits
  the earlier finding that male attractiveness leans harder on geometry — geometry
  the *right* shape features can finally capture.

## Behaviour vs. the benchmark CNNs (out-of-distribution test)

As a sanity check on how Odin behaves on faces it never trained on, I scored 60
faces — 10 attractive / 10 average / 10 unattractive per sex, from the **Chicago
Face Database** (average/unattractive) and web photos (attractive) — with Odin
(raw prediction, no boost) **and** the three SCUT benchmark CNNs run via OpenCV's
Caffe importer (AlexNet, ResNet-18, ResNeXt-50 — the ~0.81-R² models). All on the
1–10 scale (`benchmark_faces.py`).

- **All four models rank the tiers correctly** (attractive > average >
  unattractive, both sexes). Odin generalises sensibly off its SCUT training
  distribution.
- **Odin discriminates the tiers *more* than the CNNs.** It separates the
  attractive vs. unattractive means by **+2.50**, against **+1.78–1.89** for the
  CNNs, and its overall spread is wider (range **5.6 / std 1.34** vs ~4.0 / ~1.0).
  The CNNs compress toward the middle on unfamiliar faces (tail compression);
  Odin keeps using the low end.
- **Odin is systematically harsher, and the extra range is all at the bottom:**
  the CNNs floor around 3.8–4.5, Odin goes down to 2.6. Whether that low end is
  "right" is unknowable without true ratings — Odin may simply be over-harsh.

Caveats: n = 10/tier, the tiers are my own binning (not rated), and it's all
out-of-distribution — suggestive behaviour, not a benchmark. One confound worth
noting: the attractive set were small web images while the CFD faces are all
4.2 MP, so score correlates with image size (r ≈ −0.7 for *every* model) — but
that's the dataset (attractiveness and resolution are inseparable here), not a
resolution effect on the models.

## Importing the model into another project

The `.joblib` bundle is **not** self-contained: a prediction is only valid if the
51 input features are produced by this exact pipeline (same landmarks, pixel
scaling, ratio definitions, shape basis, and column order). The shape step still
computes all 25 PLS axes; the bundle's `feature_names` keeps only the named ones,
so unnamed columns are dropped on reindex. Copy the feature code with the model —
`Odin/` (the whole package) + `models/` — not just the bundle.

**Each bundle is a dict:** `xgboost`, `feature_names` (required column order),
`label`, `target`, `n_samples`, male calibration stats `xgb_pred_mean` /
`xgb_pred_max`, and the shape model `shape_ref_mean` / `shape_pls` (a fitted
`PLSRegression`; older bundles carry `shape_pca` instead).

**Minimal usage** (from the project root):
```python
import joblib, pandas as pd
from Odin.Face_analysis.landmarks import calculate_landmarks_array
from Odin.Face_analysis.face_data import extract_face_data
from Odin.Face_analysis.Ratios.appearance import appearance_features
from Odin.main import build_features, add_shape_features, male_boost

landmarks, img = calculate_landmarks_array("photo.jpg")
h, w = img.shape[:2]
px = landmarks.copy(); px[:, 0] *= w; px[:, 1] *= h; px[:, 2] *= w  # pixel space

model = joblib.load("models/model_male.joblib")
features = build_features(extract_face_data(px), appearance_features(img, px))
add_shape_features(features, px, model)               # adds the Procrustes features
X = pd.DataFrame([features], columns=model["feature_names"])

score = float(model["xgboost"].predict(X)[0])
if model["label"] == "MALE":                          # male-only presentation boost
    score = male_boost(score, model)
print(round(score, 2))
```

## License

See `LICENSE`. Note that a licence governs reuse of this **code**; the underlying
ratios are drawn from public facial-aesthetics literature and are not themselves
owned by this project.
