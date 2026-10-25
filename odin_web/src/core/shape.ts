/**
 * Port of Odin/Face_analysis/Ratios/shape.py: Procrustes-align the 478-point
 * face onto the stored mean shape, then averageness (distance to the mean) and
 * the kept PLS shape axes (sklearn PLSRegression.transform).
 */
import type { Vec } from './faceData'

export interface ShapeModel {
  refMean: number[]                 // 478*2, row-major (x0, y0, x1, y1, ...)
  xMean: number[]
  xStd: number[]
  components: Record<string, number[]>   // shape_pls_NN -> x_rotations_ column
}

export function shapeFeatures(lm: Vec[], model: ShapeModel): Record<string, number> {
  const n = lm.length
  // centre at the centroid, scale to unit centroid size
  let cx = 0, cy = 0
  for (const p of lm) { cx += p[0]; cy += p[1] }
  cx /= n; cy /= n
  const c = new Float64Array(2 * n)
  let ss = 0
  lm.forEach((p, i) => {
    c[2 * i] = p[0] - cx
    c[2 * i + 1] = p[1] - cy
    ss += c[2 * i] ** 2 + c[2 * i + 1] ** 2
  })
  const size = Math.sqrt(ss)
  if (size) for (let i = 0; i < c.length; i++) c[i] /= size

  // scipy orthogonal_procrustes(c, ref): R = U V^T of M = c^T ref, i.e. the
  // orthogonal polar factor of M = M (M^T M)^(-1/2) (closed form for 2x2).
  const ref = model.refMean
  let m00 = 0, m01 = 0, m10 = 0, m11 = 0
  for (let i = 0; i < n; i++) {
    const x = c[2 * i], y = c[2 * i + 1], rx = ref[2 * i], ry = ref[2 * i + 1]
    m00 += x * rx; m01 += x * ry; m10 += y * rx; m11 += y * ry
  }
  // S = M^T M, sqrt(S) = (S + sqrt(det S) I) / sqrt(tr S + 2 sqrt(det S))
  const s00 = m00 * m00 + m10 * m10, s01 = m00 * m01 + m10 * m11, s11 = m01 * m01 + m11 * m11
  const sd = Math.sqrt(s00 * s11 - s01 * s01)
  const t = Math.sqrt(s00 + s11 + 2 * sd)
  const q00 = (s00 + sd) / t, q01 = s01 / t, q11 = (s11 + sd) / t
  const qdet = q00 * q11 - q01 * q01
  const i00 = q11 / qdet, i01 = -q01 / qdet, i11 = q00 / qdet
  const r00 = m00 * i00 + m01 * i01, r01 = m00 * i01 + m01 * i11
  const r10 = m10 * i00 + m11 * i01, r11 = m10 * i01 + m11 * i11

  const aligned = new Float64Array(2 * n)
  let dist = 0
  for (let i = 0; i < n; i++) {
    const x = c[2 * i], y = c[2 * i + 1]
    aligned[2 * i] = x * r00 + y * r10
    aligned[2 * i + 1] = x * r01 + y * r11
    dist += (aligned[2 * i] - ref[2 * i]) ** 2 + (aligned[2 * i + 1] - ref[2 * i + 1]) ** 2
  }

  const out: Record<string, number> = { averageness: Math.sqrt(dist) }
  for (const [name, rot] of Object.entries(model.components)) {
    let s = 0
    for (let j = 0; j < aligned.length; j++) s += ((aligned[j] - model.xMean[j]) / model.xStd[j]) * rot[j]
    out[name] = s
  }
  return out
}
