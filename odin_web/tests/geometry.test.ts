// Invariances the features are designed to have: the geometric ratios don't
// depend on where the face is or how big it is, and the Procrustes shape
// features don't change under translation, scaling or in-plane rotation.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { extractFaceData, type Vec } from '../src/core/faceData'
import type { OdinModel } from '../src/core/model'
import { buildFeatures, toPixelSpace } from '../src/core/pipeline'
import { shapeFeatures } from '../src/core/shape'

const fixtures = JSON.parse(readFileSync(new URL('../parity/fixtures_ci/fixtures.json', import.meta.url), 'utf8'))
const models: Record<string, OdinModel> = Object.fromEntries(['male', 'female'].map((s) =>
  [s, JSON.parse(readFileSync(new URL(`../public/models/model_${s}.json`, import.meta.url), 'utf8'))]))

const faces = fixtures.map((f: { id: string; sex: string; landmarks: number[][]; width: number; height: number }) =>
  ({ id: f.id, sex: f.sex, px: toPixelSpace(f.landmarks, f.width, f.height) }))

// geometry only: appearance features need pixels
const ratios = (px: Vec[]) => buildFeatures(extractFaceData(px), {})

function similarity(px: Vec[], scale: number, angleDeg: number, dx: number, dy: number): Vec[] {
  const a = (angleDeg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a)
  return px.map(([x, y, z]) => [scale * (c * x - s * y) + dx, scale * (s * x + c * y) + dy, scale * z])
}

describe('geometric ratios', () => {
  it.each(faces.map((f: { id: string }) => [f.id, f]))('%s: invariant to translation and scale', (_, f) => {
    const base = ratios(f.px)
    const moved = ratios(similarity(f.px, 2.7, 0, 113, -41))
    for (const k of Object.keys(base)) expect(moved[k], k).toBeCloseTo(base[k], 9)
  })

  it('mirroring the face keeps the symmetric ratios', () => {
    const f = faces[0]
    const w = 2000
    const mirroredPx = f.px.map(([x, y, z]: Vec) => [w - x, y, z])
    const a = ratios(f.px), b = ratios(mirroredPx)
    // left/right landmarks swap roles, so averaged/symmetric measures are unchanged
    for (const k of ['width_ratio_46', 'fwhr', 'upper_third', 'middle_third', 'lower_third',
      'face_golden_ratio', 'lip_vermilion_ratio', 'lower_third_upper_split']) {
      expect(b[k], k).toBeCloseTo(a[k], 9)
    }
  })
})

describe('Procrustes shape features', () => {
  it.each(faces.map((f: { id: string }) => [f.id, f]))('%s: invariant to translation, scale and rotation', (_, f) => {
    const model = models[f.sex].shape
    const base = shapeFeatures(f.px, model)
    const moved = shapeFeatures(similarity(f.px, 0.4, 23, -500, 77), model)
    for (const k of Object.keys(base)) expect(moved[k], k).toBeCloseTo(base[k], 9)
  })

  it('the mean shape itself has averageness 0', () => {
    for (const m of Object.values(models)) {
      const ref = m.shape.refMean
      const px: Vec[] = Array.from({ length: ref.length / 2 }, (_, i) => [ref[2 * i] * 900 + 300, ref[2 * i + 1] * 900 + 200, 0])
      expect(shapeFeatures(px, m.shape).averageness).toBeLessThan(1e-12)
    }
  })
})
