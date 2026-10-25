/**
 * The handful of OpenCV operations the Odin pipeline uses, reimplemented so the
 * browser build reproduces the Python numbers:
 *
 *   cvtColor BGR2LAB / BGR2GRAY (8-bit) - bit-exact with OpenCV 4.13's integer
 *                                         paths (checked on all 2^24 colours)
 *   fillPoly, convexHull                - pixel-exact masks (8-connected outline
 *                                         + even-odd scanline fill)
 *   Laplacian (ksize 1), GaussianBlur   - float64, BORDER_REFLECT_101
 *
 * Images are RGBA (what canvas/ImageData gives); OpenCV's BGR order is only a
 * memory layout, so the channel maths is identical.
 */

export interface RgbaImage {
  width: number
  height: number
  data: Uint8ClampedArray | Uint8Array   // RGBA, row-major
}

export type Polygon = [number, number][]  // integer pixel vertices

/** numpy/cvRound-style rounding: halves go to the even neighbour. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x)
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r
}

// ---------------------------------------------------------------- colour ----

// OpenCV RGB2Lab_b: sRGB gamma table (x8), fixed-point XYZ, cube-root table.
const GAMMA_SHIFT = 3
const LAB_SHIFT = 12
const LAB_SHIFT2 = 15
const GAMMA_TAB = new Int32Array(256)
const CBRT_TAB = new Int32Array(256 * 3 / 2 * (1 << GAMMA_SHIFT))
const LAB_C: number[] = []
const L_SCALE = Math.floor((116 * 255 + 50) / 100)
const L_SHIFT = -Math.floor((16 * 255 * (1 << LAB_SHIFT2) + 50) / 100)
{
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    const g = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4)
    GAMMA_TAB[i] = Math.min(65535, roundHalfEven(255 * (1 << GAMMA_SHIFT) * g))
  }
  for (let i = 0; i < CBRT_TAB.length; i++) {
    const x = i / (255 * (1 << GAMMA_SHIFT))
    const c = x < 0.008856 ? x * 7.787 + 0.13793103448275862 : Math.cbrt(x)
    // OpenCV builds this table in float32: round the float32 value, not the double.
    CBRT_TAB[i] = Math.min(65535, roundHalfEven(Math.fround((1 << LAB_SHIFT2) * c)))
  }
  const M = [0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169,
             0.019334, 0.119193, 0.950227]
  const WHITE = [0.950456, 1.0, 1.088754]
  for (let i = 0; i < 9; i++) LAB_C.push(roundHalfEven((1 << LAB_SHIFT) * M[i] / WHITE[Math.floor(i / 3)]))
}

const descale = (x: number, n: number) => (x + (1 << (n - 1))) >> n
const sat8 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v)

/** One 8-bit RGB pixel -> OpenCV 8-bit Lab (L*255/100, a+128, b+128). */
export function rgbToLab8(r: number, g: number, b: number): [number, number, number] {
  const R = GAMMA_TAB[r], G = GAMMA_TAB[g], B = GAMMA_TAB[b]
  const C = LAB_C
  const fX = CBRT_TAB[descale(R * C[0] + G * C[1] + B * C[2], LAB_SHIFT)]
  const fY = CBRT_TAB[descale(R * C[3] + G * C[4] + B * C[5], LAB_SHIFT)]
  const fZ = CBRT_TAB[descale(R * C[6] + G * C[7] + B * C[8], LAB_SHIFT)]
  return [
    sat8(descale(L_SCALE * fY + L_SHIFT, LAB_SHIFT2)),
    sat8(descale(500 * (fX - fY) + 128 * (1 << LAB_SHIFT2), LAB_SHIFT2)),
    sat8(descale(200 * (fY - fZ) + 128 * (1 << LAB_SHIFT2), LAB_SHIFT2)),
  ]
}

/** Whole image -> interleaved 8-bit Lab (3 bytes per pixel). */
export function toLab(img: RgbaImage): Uint8Array {
  const n = img.width * img.height
  const out = new Uint8Array(n * 3)
  const d = img.data
  for (let i = 0; i < n; i++) {
    const lab = rgbToLab8(d[4 * i], d[4 * i + 1], d[4 * i + 2])
    out[3 * i] = lab[0]
    out[3 * i + 1] = lab[1]
    out[3 * i + 2] = lab[2]
  }
  return out
}

/** OpenCV BGR2GRAY (8-bit): 15-bit fixed-point BT.601 weights. */
export function toGray(img: RgbaImage): Uint8Array {
  const n = img.width * img.height
  const out = new Uint8Array(n)
  const d = img.data
  for (let i = 0; i < n; i++) {
    out[i] = (d[4 * i + 2] * 3735 + d[4 * i + 1] * 19235 + d[4 * i] * 9798 + 16384) >> 15
  }
  return out
}

// ------------------------------------------------------------- polygons ----

const XY_SHIFT = 16
const XY_ONE = 1 << XY_SHIFT

/** cv2.line(..., LINE_8) pixels via OpenCV's LineIterator (left-to-right). */
function drawLine8(mask: Uint8Array, w: number, h: number,
                   x0: number, y0: number, x1: number, y1: number) {
  if (x1 < x0) { [x0, x1] = [x1, x0]; [y0, y1] = [y1, y0] }
  const dx = x1 - x0
  const ady = Math.abs(y1 - y0)
  const ystep = y1 < y0 ? -1 : 1
  const steep = ady > dx
  const major = steep ? ady : dx
  const minor = steep ? dx : ady
  let err = major - 2 * minor
  let x = x0, y = y0
  for (let i = 0; i <= major; i++) {
    if (x >= 0 && x < w && y >= 0 && y < h) mask[y * w + x] = 1
    const take = err < 0
    err += -2 * minor + (take ? 2 * major : 0)
    if (steep) { y += ystep; if (take) x += 1 }
    else { x += 1; if (take) y += ystep }
  }
}

/**
 * cv2.fillPoly(mask, polygons, 255) for non-antialiased integer polygons:
 * every edge is drawn as an 8-connected line, then scanlines are filled
 * between sorted edge crossings (even-odd across all polygons, like OpenCV's
 * single edge collection). Returns a 0/1 mask.
 */
export function fillPolyMask(w: number, h: number, polygons: Polygon[]): Uint8Array {
  const mask = new Uint8Array(w * h)
  const edges: { y0: number; y1: number; x: number; dx: number }[] = []
  for (const poly of polygons) {
    const n = poly.length
    for (let i = 0; i < n; i++) {
      const [ax, ay] = poly[(i + n - 1) % n]
      const [bx, by] = poly[i]
      drawLine8(mask, w, h, ax, ay, bx, by)
      if (ay === by) continue
      const fx0 = ax * XY_ONE, fx1 = bx * XY_ONE
      const dx = Math.trunc((fx1 - fx0) / (by - ay))       // C integer division
      if (ay < by) edges.push({ y0: ay, y1: by, x: fx0, dx })
      else edges.push({ y0: by, y1: ay, x: fx1, dx })
    }
  }
  if (edges.length < 2) return mask
  let yMin = Infinity, yMax = -Infinity
  for (const e of edges) { yMin = Math.min(yMin, e.y0); yMax = Math.max(yMax, e.y1) }
  yMax = Math.min(yMax, h)
  const xs: number[] = []
  for (let y = Math.max(0, yMin); y < yMax; y++) {
    xs.length = 0
    for (const e of edges) if (e.y0 <= y && y < e.y1) xs.push(e.x + (y - e.y0) * e.dx)
    xs.sort((a, b) => a - b)
    for (let k = 0; k + 1 < xs.length; k += 2) {
      // left crossing rounds up, right rounds down (OpenCV's non-AA rule)
      let x1 = Math.floor((xs[k] + XY_ONE - 1) / XY_ONE)
      let x2 = Math.floor(xs[k + 1] / XY_ONE)
      if (x1 >= w || x2 < 0) continue
      if (x1 < 0) x1 = 0
      if (x2 >= w) x2 = w - 1
      mask.fill(1, y * w + x1, y * w + x2 + 1)
    }
  }
  return mask
}

/** Convex hull of integer points (monotone chain), as a closed vertex loop. */
export function convexHull(points: [number, number][]): Polygon {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  if (pts.length < 3) return pts
  const cross = (o: number[], a: number[], b: number[]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const lower: [number, number][] = []
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop()
    lower.push(p)
  }
  const upper: [number, number][] = []
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop()
    upper.push(p)
  }
  lower.pop()
  upper.pop()
  return lower.concat(upper)
}

// -------------------------------------------------------------- filters ----

/** OpenCV borderInterpolate for BORDER_REFLECT_101 (handles multiple bounces). */
export function reflect101(p: number, len: number): number {
  if (p >= 0 && p < len) return p
  if (len === 1) return 0
  do {
    p = p < 0 ? -p : len - 1 - (p - len) - 1
  } while (p < 0 || p >= len)
  return p
}

/** cv2.Laplacian(gray, CV_64F) (ksize 1: 4-neighbour kernel) at one pixel. */
export function laplacianAt(gray: Uint8Array, w: number, h: number, x: number, y: number): number {
  const row = y * w
  const up = reflect101(y - 1, h) * w
  const dn = reflect101(y + 1, h) * w
  return gray[up + x] + gray[dn + x] + gray[row + reflect101(x - 1, w)] +
    gray[row + reflect101(x + 1, w)] - 4 * gray[row + x]
}

/** OpenCV getGaussianKernel for a float64 image (ksize derived from sigma). */
export function gaussianKernel(sigma: number): Float64Array {
  const n = roundHalfEven(sigma * 4 * 2 + 1) | 1
  const k = new Float64Array(n)
  const scale = -0.5 / (sigma * sigma)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const x = i - (n - 1) * 0.5
    k[i] = Math.exp(scale * x * x)
    sum += k[i]
  }
  for (let i = 0; i < n; i++) k[i] /= sum
  return k
}

/**
 * cv2.GaussianBlur(src, (0, 0), sigma) on a float64 single-channel image,
 * evaluated only at the given pixel indices (the blur is separable, so only
 * the rows those pixels reach are row-filtered). Much cheaper than a full
 * blur on a multi-megapixel photo when only a cheek patch is needed.
 */
export function gaussianBlurAt(src: Float64Array, w: number, h: number,
                               sigma: number, pixels: Int32Array): Float64Array {
  const k = gaussianKernel(sigma)
  const r = (k.length - 1) / 2
  let x0 = w, x1 = -1
  for (const p of pixels) { const x = p % w; if (x < x0) x0 = x; if (x > x1) x1 = x }
  const bw = x1 - x0 + 1
  const rows = new Map<number, Float64Array>()
  const rowPass = (yy: number) => {
    let row = rows.get(yy)
    if (row) return row
    row = new Float64Array(bw)
    const base = yy * w
    for (let x = x0; x <= x1; x++) {
      let s = 0
      for (let j = 0; j < k.length; j++) s += src[base + reflect101(x + j - r, w)] * k[j]
      row[x - x0] = s
    }
    rows.set(yy, row)
    return row
  }
  const out = new Float64Array(pixels.length)
  pixels.forEach((p, i) => {
    const x = p % w, y = (p - x) / w
    let s = 0
    for (let j = 0; j < k.length; j++) s += rowPass(reflect101(y + j - r, h))[x - x0] * k[j]
    out[i] = s
  })
  return out
}

/** Indices of the set pixels of a 0/1 mask, in row-major order. */
export function maskPixels(mask: Uint8Array): Int32Array {
  let n = 0
  for (let i = 0; i < mask.length; i++) n += mask[i]
  const out = new Int32Array(n)
  let j = 0
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[j++] = i
  return out
}
