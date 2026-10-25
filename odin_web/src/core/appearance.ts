/**
 * Port of Odin/Face_analysis/Ratios/regions.py + appearance.py: sampling
 * polygons, then skin texture / colour-unevenness / blemish, mean colours and
 * the CIELab facial-contrast features. Missing values are NaN (pandas None).
 */
import type { Vec } from './faceData'
import {
  convexHull, fillPolyMask, gaussianBlurAt, laplacianAt, maskPixels, rgbToLab8,
  type Polygon, type RgbaImage,
} from './imageOps'

const LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
const LEFT_IRIS = [469, 470, 471, 472]
const RIGHT_IRIS = [474, 475, 476, 477]
const LEFT_CHEEK = [116, 123, 111, 117, 118, 119, 50, 101, 36, 205, 206, 207, 187, 147]
const RIGHT_CHEEK = [345, 352, 340, 346, 347, 348, 280, 330, 266, 425, 426, 427, 411, 376]

const IRIS_DARK_LUMA = 30

export type Regions = Record<'lips' | 'left_iris' | 'right_iris' | 'left_cheek' | 'right_cheek', Polygon>

function poly(lm: Vec[], idxs: number[], hull = false): Polygon {
  // np.array(..., dtype=np.int32) truncates toward zero
  const pts = idxs.map((i) => [Math.trunc(lm[i][0]), Math.trunc(lm[i][1])] as [number, number])
  return hull ? convexHull(pts) : pts
}

export function extractRegions(lm: Vec[]): Regions {
  return {
    lips: poly(lm, LIPS_OUTER),
    left_iris: poly(lm, LEFT_IRIS, true),
    right_iris: poly(lm, RIGHT_IRIS, true),
    left_cheek: poly(lm, LEFT_CHEEK, true),
    right_cheek: poly(lm, RIGHT_CHEEK, true),
  }
}

const mean = (v: ArrayLike<number>) => {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i]
  return v.length ? s / v.length : NaN
}
/** numpy .var() / .std() (population, ddof=0). */
const variance = (v: ArrayLike<number>) => {
  const m = mean(v)
  let s = 0
  for (let i = 0; i < v.length; i++) s += (v[i] - m) ** 2
  return v.length ? s / v.length : NaN
}

export interface ImageCache {
  img: RgbaImage
  lab: Uint8Array
  gray: Uint8Array
}

function laplacianTexture(c: ImageCache, polygon: Polygon): number {
  const { width: w, height: h } = c.img
  const px = maskPixels(fillPolyMask(w, h, [polygon]))
  const vals = new Float64Array(px.length)
  px.forEach((p, i) => { vals[i] = laplacianAt(c.gray, w, h, p % w, Math.floor(p / w)) })
  return variance(vals)
}

function meanColor(c: ImageCache, polygons: Polygon[], darkLuma?: number): number[] | null {
  const d = c.img.data
  let r = 0, g = 0, b = 0, n = 0
  for (const p of maskPixels(fillPolyMask(c.img.width, c.img.height, polygons))) {
    const R = d[4 * p], G = d[4 * p + 1], B = d[4 * p + 2]
    if (darkLuma !== undefined && B * 0.114 + G * 0.587 + R * 0.299 < darkLuma) continue
    r += R; g += G; b += B; n++
  }
  return n ? [r / n, g / n, b / n] : null
}

/** Mean colour -> standard CIELab via OpenCV's 8-bit conversion (uint8 cast truncates). */
function rgbToLab(rgb: number[] | null): number[] {
  if (!rgb) return [NaN, NaN, NaN]
  const [L, a, b] = rgbToLab8(Math.trunc(rgb[0]), Math.trunc(rgb[1]), Math.trunc(rgb[2]))
  return [L * 100 / 255, a - 128, b - 128]
}

export function appearanceFeatures(c: ImageCache, lm: Vec[], reg: Regions): Record<string, number> {
  const { width: w, height: h } = c.img
  const cheeks = [reg.left_cheek, reg.right_cheek]

  // nanmean over the two cheeks
  const tex = [laplacianTexture(c, reg.left_cheek), laplacianTexture(c, reg.right_cheek)]
    .filter((v) => !Number.isNaN(v))
  const skinTexture = tex.length ? mean(tex) : NaN

  // a*/b* spread and the a* difference-of-Gaussians blemish score, both cheeks
  const cheekPx = maskPixels(fillPolyMask(w, h, cheeks))
  const aVals = new Float64Array(cheekPx.length)
  const bVals = new Float64Array(cheekPx.length)
  cheekPx.forEach((p, i) => { aVals[i] = c.lab[3 * p + 1]; bVals[i] = c.lab[3 * p + 2] })
  const skinAStd = Math.sqrt(variance(aVals))
  const skinBStd = Math.sqrt(variance(bVals))

  const faceWidth = Math.abs(lm[454][0] - lm[234][0])
  const sSmall = Math.max(1.0, 0.006 * faceWidth)
  const sLarge = Math.max(sSmall * 3.0, 0.02 * faceWidth)
  const aChan = new Float64Array(w * h)
  for (let i = 0; i < w * h; i++) aChan[i] = c.lab[3 * i + 1]
  let spot = 0
  if (cheekPx.length) {
    const small = gaussianBlurAt(aChan, w, h, sSmall, cheekPx)
    const large = gaussianBlurAt(aChan, w, h, sLarge, cheekPx)
    let s = 0, n = 0
    for (let i = 0; i < cheekPx.length; i++) {
      const dog = small[i] - large[i]
      if (dog > 0) { s += dog; n++ }
    }
    spot = n ? s / n : 0
  }

  const lips = meanColor(c, [reg.lips])
  const eye = meanColor(c, [reg.left_iris, reg.right_iris], IRIS_DARK_LUMA)
  const skin = meanColor(c, cheeks)
  const [lipsL, lipsA] = rgbToLab(lips)
  const [eyeL, eyeA] = rgbToLab(eye)
  const [skinL, skinA] = rgbToLab(skin)
  const eyeSkinLum = Math.abs(eyeL - skinL)
  const lipSkinLum = Math.abs(lipsL - skinL)
  const ch = (rgb: number[] | null, i: number) => (rgb ? rgb[i] : NaN)

  return {
    skin_texture: skinTexture,
    skin_a_std: skinAStd,
    skin_b_std: skinBStd,
    skin_spot_burden: spot,
    lips_r: ch(lips, 0), lips_g: ch(lips, 1), lips_b: ch(lips, 2),
    eye_r: ch(eye, 0), eye_g: ch(eye, 1), eye_b: ch(eye, 2),
    skin_r: ch(skin, 0), skin_g: ch(skin, 1), skin_b: ch(skin, 2),
    eye_skin_luminance_contrast: eyeSkinLum,
    lip_skin_luminance_contrast: lipSkinLum,
    lip_skin_redness_contrast: Math.abs(lipsA - skinA),
    eye_skin_redness_contrast: Math.abs(eyeA - skinA),
    facial_contrast_avg: (eyeSkinLum + lipSkinLum) / 2,
  }
}
