# Odin — in-browser build

The whole Odin pipeline running client-side, so it can be hosted on GitHub Pages
as a static site: MediaPipe's FaceLandmarker (WASM) → the hairline detection, 25
geometric ratios, colour/texture features and Procrustes/PLS shape features
(ported to TypeScript) → the XGBoost model and exact TreeSHAP contributions.
The photo is never uploaded. Visitors without a photo can click one of four
sample faces in `public/samples/`: AI-generated (StyleGAN), so no real person is
rated, and screened to be frontal with a neutral expression like SCUT's photos.
Their sources are listed in `src/samples.ts`.

Because the ratios and shape features assume a frontal, neutral photo like SCUT's,
the site also reads MediaPipe's head pose and blendshapes and warns when a photo is
outside the training range: head turned > 10°, tilted sideways > 8°, tilted up/down
outside −10…22°, or mouth open (thresholds and the SCUT distribution they come from
are in `src/core/quality.ts`; a smile only gets a soft note, since SCUT has plenty).

## Parity with the Python pipeline

The port has to produce the same numbers as the model was trained on, so it is
tested stage by stage against the Python reference:

| Check | How | Result (36 SCUT faces, incl. 4 resized to 760×620) |
|---|---|---|
| Ported feature code + model | `npm run parity` (Node, using Python's own landmarks + pixels) | all 51 features, score and every SHAP value within 1e-6 (max score Δ 7.6e-7, from XGBoost's float32 sums) |
| MediaPipe WASM vs. Python | `npm run parity:browser` (headless Chrome) | landmark drift 0.0007 px mean / 0.019 px max; 35/36 scores identical, one off by 0.027 |
| Real JPEG, end to end | `node scripts/smoke.mjs photo.jpg female` | same score as `python -m Odin.main` |

Getting there meant matching OpenCV exactly where it matters: the 8-bit
`BGR2LAB` and `BGR2GRAY` integer paths (verified bit-exact on all 2^24 colours),
`fillPoly`'s rasterisation rule (pixel-exact masks), `GaussianBlur`'s kernel size
and `BORDER_REFLECT_101`, and Python's round-half-to-even in the hairline walk.

## Layout

```
export_model.py        joblib bundles → public/models/*.json (+ copies face_landmarker.task)
src/core/              DOM-free port of the pipeline (runs in browser and Node)
  imageOps.ts          the OpenCV operations used (Lab, gray, fillPoly, hull, Laplacian, Gaussian)
  faceData.ts ratios.ts trichion.ts appearance.ts shape.ts
  model.ts             XGBoost inference + TreeSHAP
  pipeline.ts          landmarks + image → features → score/contributions
  quality.ts           head pose / expression checks behind the photo warnings
  present.ts           port of odin_ui/backend/app.py's response building
src/api.ts             browser glue: decode, MediaPipe, analyze()
src/App.tsx …          the odin_ui React frontend, unchanged apart from api.ts
parity/                make_fixtures.py (Python) + parity.ts (Node); fixtures/ is SCUT data — never commit
parity.html            dev-only browser landmark parity page
scripts/               copy-wasm, browser-parity, smoke
```

## Usage

```bash
# from the repo root, once (and after retraining)
venv/Scripts/python.exe odin_web/export_model.py

cd odin_web
npm install
npm run dev          # http://localhost:5173
npm run build        # static site in dist/ (~27 MB, mostly the MediaPipe WASM)

# parity (needs odin_training/scut/)
../venv/Scripts/python.exe parity/make_fixtures.py
npm run parity
npm run parity:browser
```

## Deploying to GitHub Pages

`npm run deploy` builds and pushes `dist/` to the repo's `gh-pages` branch; then
in the repo's *Settings → Pages* choose "Deploy from a branch" → `gh-pages`. The
site will be at `https://kawasakicode.github.io/Odin/` (the build uses relative
paths, so any sub-path works).

Not ported: the SCUT benchmark CNN scores (third-party Caffe weights, not
redistributed) and the male presentation boost (the UI shows the raw score, as
the Python UI does; `maleBoost` exists in `pipeline.ts` if wanted).
