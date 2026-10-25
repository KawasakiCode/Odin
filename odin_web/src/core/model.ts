/**
 * XGBoost inference + exact TreeSHAP contributions over the exported trees
 * (export_model.py), matching booster.predict(..., pred_contribs=True):
 * contributions sum to the prediction, the bias is base_score plus each tree's
 * cover-weighted mean. Features are compared as float32 like XGBoost does, and
 * NaN follows each split's default direction.
 */
import type { ShapeModel } from './shape'

export interface Tree {
  left: number[]
  right: number[]
  feature: number[]
  value: number[]          // split threshold, or the leaf value on leaves
  defaultLeft: number[]
  cover: number[]
}

export interface OdinModel {
  label: string
  featureNames: string[]
  baseScore: number
  trees: Tree[]
  xgbPredMean: number | null
  xgbPredMax: number | null
  averagenessQuantiles: number[]
  shape: ShapeModel
}

const isLeaf = (t: Tree, i: number) => t.left[i] === -1

function next(t: Tree, i: number, x: Float32Array): number {
  const v = x[t.feature[i]]
  if (Number.isNaN(v)) return t.defaultLeft[i] ? t.left[i] : t.right[i]
  return v < Math.fround(t.value[i]) ? t.left[i] : t.right[i]
}

export function predict(model: OdinModel, features: number[]): number {
  const x = Float32Array.from(features)
  let s = 0
  for (const t of model.trees) {
    let i = 0
    while (!isLeaf(t, i)) i = next(t, i, x)
    s += Math.fround(t.value[i])
  }
  return s + model.baseScore
}

// ---- TreeSHAP (Lundberg et al., Algorithm 2), as in XGBoost's TreeShap ----

interface PathElem { d: number; z: number; o: number; w: number }

function extend(m: PathElem[], pz: number, po: number, pi: number): PathElem[] {
  const l = m.length
  const out = m.map((e) => ({ ...e }))
  out.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 })
  for (let i = l - 1; i >= 0; i--) {
    out[i + 1].w += po * out[i].w * (i + 1) / (l + 1)
    out[i].w = pz * out[i].w * (l - i) / (l + 1)
  }
  return out
}

function unwind(m: PathElem[], i: number): PathElem[] {
  const l = m.length - 1
  const out = m.map((e) => ({ ...e }))
  let n = out[l].w
  const { o, z } = out[i]
  for (let j = l - 1; j >= 0; j--) {
    if (o !== 0) {
      const t = out[j].w
      out[j].w = n * (l + 1) / ((j + 1) * o)
      n = t - out[j].w * z * (l - j) / (l + 1)
    } else {
      out[j].w = out[j].w * (l + 1) / (z * (l - j))
    }
  }
  for (let j = i; j < l; j++) {
    out[j].d = out[j + 1].d
    out[j].z = out[j + 1].z
    out[j].o = out[j + 1].o
  }
  out.pop()
  return out
}

function unwoundSum(m: PathElem[], i: number): number {
  const l = m.length - 1
  const { o, z } = m[i]
  let n = m[l].w
  let total = 0
  for (let j = l - 1; j >= 0; j--) {
    if (o !== 0) {
      const t = n * (l + 1) / ((j + 1) * o)
      total += t
      n = m[j].w - t * z * (l - j) / (l + 1)
    } else {
      total += (m[j].w / z) / ((l - j) / (l + 1))
    }
  }
  return total
}

function treeShap(t: Tree, x: Float32Array, phi: Float64Array, node: number,
                  path: PathElem[], pz: number, po: number, pi: number) {
  const m = extend(path, pz, po, pi)
  if (isLeaf(t, node)) {
    const v = Math.fround(t.value[node])
    for (let i = 1; i < m.length; i++) {
      phi[m[i].d] += unwoundSum(m, i) * (m[i].o - m[i].z) * v
    }
    return
  }
  const hot = next(t, node, x)
  const cold = hot === t.left[node] ? t.right[node] : t.left[node]
  const d = t.feature[node]
  let iz = 1, io = 1
  let mm = m
  const k = m.findIndex((e, idx) => idx > 0 && e.d === d)
  if (k > 0) {
    iz = m[k].z
    io = m[k].o
    mm = unwind(m, k)
  }
  const c = t.cover[node]
  treeShap(t, x, phi, hot, mm, iz * t.cover[hot] / c, io, d)
  treeShap(t, x, phi, cold, mm, iz * t.cover[cold] / c, 0, d)
}

function meanValue(t: Tree, i: number): number {
  if (isLeaf(t, i)) return Math.fround(t.value[i])
  const l = t.left[i], r = t.right[i]
  return (meanValue(t, l) * t.cover[l] + meanValue(t, r) * t.cover[r]) / t.cover[i]
}

/** Per-feature contributions (in featureNames order) and the bias term. */
export function contributions(model: OdinModel, features: number[]) {
  const x = Float32Array.from(features)
  const phi = new Float64Array(features.length)
  let base = model.baseScore
  for (const t of model.trees) {
    treeShap(t, x, phi, 0, [], 1, 1, -1)
    base += meanValue(t, 0)
  }
  return { base, phi: Array.from(phi) }
}
