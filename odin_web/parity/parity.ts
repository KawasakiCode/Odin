/**
 * Parity test: run the TypeScript pipeline on the fixtures made by
 * make_fixtures.py and compare with the Python reference, stage by stage.
 * Uses Python's own MediaPipe landmarks, so this isolates the ported code
 * (ratios, hairline, regions, colour/texture, shape, XGBoost, TreeSHAP) from
 * any browser-vs-Python landmark drift (that part is checked in parity.html).
 *
 *   npm run parity
 */
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { extractRegions } from '../src/core/appearance'
import { fillPolyMask, maskPixels } from '../src/core/imageOps'
import { contributions, predict, type OdinModel } from '../src/core/model'
import { analyzeFace, toPixelSpace } from '../src/core/pipeline'

const HERE = dirname(fileURLToPath(import.meta.url))
const FIX = join(HERE, 'fixtures')
const MODELS = join(HERE, '..', 'public', 'models')

interface Fixture {
  id: string
  sex: 'male' | 'female'
  width: number
  height: number
  landmarks: number[][]
  forehead: number[]
  regions: Record<string, [number, number][]>
  maskCounts: Record<string, number>
  features: Record<string, number | null>
  score: number
  contribs: number[]
  base: number
}

const fixtures: Fixture[] = JSON.parse(readFileSync(join(FIX, 'fixtures.json'), 'utf8'))
const models: Record<string, OdinModel> = {
  male: JSON.parse(readFileSync(join(MODELS, 'model_male.json'), 'utf8')),
  female: JSON.parse(readFileSync(join(MODELS, 'model_female.json'), 'utf8')),
}

// Relative error with an absolute floor, so near-zero values don't blow up.
const err = (a: number, b: number) => Math.abs(a - b) / Math.max(1, Math.abs(b))
const worst: Record<string, { err: number; id: string; js: number; py: number }> = {}
const track = (key: string, id: string, js: number, py: number | null) => {
  const e = py == null ? (Number.isNaN(js) ? 0 : Infinity) : err(js, py)
  if (!worst[key] || e > worst[key].err) worst[key] = { err: e, id, js, py: py ?? NaN }
}

let polyMismatch = 0
for (const f of fixtures) {
  const model = models[f.sex]
  const img = {
    width: f.width,
    height: f.height,
    data: new Uint8Array(readFileSync(join(FIX, `${f.id}.rgba`))),
  }
  const px = toPixelSpace(f.landmarks, f.width, f.height)

  // 1. regions + masks
  const reg = extractRegions(px)
  for (const [k, poly] of Object.entries(reg)) {
    const key = (p: number[][]) => p.map((q) => q.join(',')).sort().join(' ')
    if (key(poly) !== key(f.regions[k])) polyMismatch++
  }
  const masks: Record<string, [number, number][][]> = {
    lips: [reg.lips], irises: [reg.left_iris, reg.right_iris],
    cheeks: [reg.left_cheek, reg.right_cheek], left_cheek: [reg.left_cheek],
  }
  for (const [k, polys] of Object.entries(masks)) {
    track(`mask:${k}`, f.id, maskPixels(fillPolyMask(f.width, f.height, polys)).length, f.maskCounts[k])
  }

  // 2. full post-landmark pipeline
  const a = analyzeFace(img, px, model)
  track('forehead.x', f.id, a.faceData.top_center_forehead[0], f.forehead[0])
  track('forehead.y', f.id, a.faceData.top_center_forehead[1], f.forehead[1])
  for (const k of model.featureNames) track(k, f.id, a.features[k], f.features[k])
  track('SCORE', f.id, a.score, f.score)
  track('SHAP base', f.id, a.base, f.base)
  model.featureNames.forEach((k, i) => track('SHAP (max over features)', f.id, a.contribs[k], f.contribs[i]))

  // 3. model alone, fed Python's exact features
  const x = model.featureNames.map((k) => f.features[k] ?? NaN)
  track('model-only score', f.id, predict(model, x), f.score)
  const c = contributions(model, x)
  model.featureNames.forEach((_, i) => track('model-only SHAP', f.id, c.phi[i], f.contribs[i]))
  track('SHAP sums to score', f.id, c.base + c.phi.reduce((s, v) => s + v, 0), f.score)
}

const TOL = 1e-6
let fails = 0
console.log(`${fixtures.length} fixtures, convex-hull/polygon mismatches: ${polyMismatch}\n`)
console.log('check'.padEnd(36), 'max rel err'.padStart(12), '  worst fixture (js vs py)')
for (const [k, w] of Object.entries(worst)) {
  const bad = w.err > TOL
  if (bad) fails++
  console.log(`${bad ? '✗' : ' '} ${k.padEnd(34)} ${w.err.toExponential(2).padStart(12)}  ` +
    (bad ? `${w.id} (${w.js} vs ${w.py})` : ''))
}
console.log(fails ? `\n${fails} checks above ${TOL}` : `\nall checks within ${TOL}`)
process.exit(fails || polyMismatch ? 1 : 0)
