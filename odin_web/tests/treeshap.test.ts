// TreeSHAP against brute-force Shapley values on random trees, plus local
// accuracy (contributions sum to the prediction) on the real exported models.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { contributions, predict, type OdinModel, type Tree } from '../src/core/model'

// small deterministic PRNG
function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32)
}

/** Random tree over `m` features; features may repeat on a path (exercises unwind). */
function randomTree(rand: () => number, m: number, depth: number): Tree {
  const t: Tree = { left: [], right: [], feature: [], value: [], defaultLeft: [], cover: [] }
  const build = (d: number): number => {
    const i = t.left.length
    t.left.push(-1); t.right.push(-1); t.feature.push(0); t.value.push(0); t.defaultLeft.push(0); t.cover.push(0)
    if (d === 0 || rand() < 0.2) {
      t.value[i] = Math.fround(rand() * 4 - 2)
      t.cover[i] = 1 + Math.floor(rand() * 20)
      return i
    }
    t.feature[i] = Math.floor(rand() * m)
    t.value[i] = Math.fround(rand())
    t.defaultLeft[i] = rand() < 0.5 ? 1 : 0
    const l = build(d - 1), r = build(d - 1)
    t.left[i] = l; t.right[i] = r
    t.cover[i] = t.cover[l] + t.cover[r]
    return i
  }
  build(depth)
  return t
}

/** Path-dependent conditional expectation E[f | x_S], the value function TreeSHAP uses. */
function expectation(t: Tree, x: Float32Array, S: Set<number>, i = 0): number {
  if (t.left[i] === -1) return Math.fround(t.value[i])
  const f = t.feature[i]
  if (S.has(f)) {
    const v = x[f]
    const goLeft = Number.isNaN(v) ? t.defaultLeft[i] === 1 : v < Math.fround(t.value[i])
    return expectation(t, x, S, goLeft ? t.left[i] : t.right[i])
  }
  const l = t.left[i], r = t.right[i]
  return (t.cover[l] * expectation(t, x, S, l) + t.cover[r] * expectation(t, x, S, r)) / t.cover[i]
}

function bruteShapley(trees: Tree[], x: Float32Array, m: number): number[] {
  const fact = (n: number): number => (n <= 1 ? 1 : n * fact(n - 1))
  const v = (S: Set<number>) => trees.reduce((s, t) => s + expectation(t, x, S), 0)
  const phi = new Array(m).fill(0)
  for (let i = 0; i < m; i++) {
    const others = [...Array(m).keys()].filter((j) => j !== i)
    for (let mask = 0; mask < 1 << others.length; mask++) {
      const S = new Set(others.filter((_, k) => mask & (1 << k)))
      const w = (fact(S.size) * fact(m - S.size - 1)) / fact(m)
      phi[i] += w * (v(new Set([...S, i])) - v(S))
    }
  }
  return phi
}

const model = (trees: Tree[], m: number, baseScore = 0.5): OdinModel => ({
  label: 'TEST', featureNames: [...Array(m).keys()].map(String), baseScore, trees,
  xgbPredMean: null, xgbPredMax: null, averagenessQuantiles: [],
  shape: { refMean: [], xMean: [], xStd: [], components: {} },
})

describe('TreeSHAP', () => {
  it('equals brute-force Shapley values on random trees (incl. repeated features and NaN)', () => {
    const rand = rng(7)
    for (let trial = 0; trial < 40; trial++) {
      const m = 2 + Math.floor(rand() * 4)                    // 2..5 features
      const trees = Array.from({ length: 1 + Math.floor(rand() * 3) }, () => randomTree(rand, m, 4))
      const x = Array.from({ length: m }, () => (rand() < 0.15 ? NaN : rand()))
      const { phi, base } = contributions(model(trees, m), x)
      const expected = bruteShapley(trees, Float32Array.from(x), m)
      phi.forEach((p, i) => expect(p).toBeCloseTo(expected[i], 9))
      // bias = base score + E[f] over the training covers
      expect(base).toBeCloseTo(0.5 + trees.reduce((s, t) => s + expectation(t, Float32Array.from(x), new Set()), 0), 9)
    }
  })

  for (const sex of ['male', 'female']) {
    it(`sums to the prediction on the real ${sex} model`, () => {
      const m: OdinModel = JSON.parse(readFileSync(new URL(`../public/models/model_${sex}.json`, import.meta.url), 'utf8'))
      const rand = rng(sex.length)
      for (let k = 0; k < 20; k++) {
        // plausible-ish random inputs, with some missing values
        const x = m.featureNames.map(() => (rand() < 0.05 ? NaN : rand() * 2))
        const { phi, base } = contributions(m, x)
        expect(base + phi.reduce((s, v) => s + v, 0)).toBeCloseTo(predict(m, x), 4)
      }
    })
  }
})
