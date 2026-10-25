/**
 * Port of Odin/Face_analysis/trichion.py: walk up the forehead midline from
 * MediaPipe's top point and stop where the colour departs from skin for a run
 * of steps (the hairline). Replaces face_data.top_center_forehead in place.
 */
import type { FaceData } from './faceData'
import { roundHalfEven, type RgbaImage } from './imageOps'

const FLOOR = 5
const K = 3

/** Mean 8-bit Lab colour of the square window around pos (+ the pixels, for stats). */
function windowLab(lab: Uint8Array, img: RgbaImage, size: number, pos: number[]) {
  let dw = Math.floor((size - 1) / 2) * 2 + 1          // round down to odd
  if (dw < 7) dw = 7
  const half = Math.floor(dw / 2)
  const cx = roundHalfEven(pos[0]), cy = roundHalfEven(pos[1])   // Python round()
  const x0 = Math.max(0, cx - half), x1 = Math.min(img.width, cx + half + 1)
  const y0 = Math.max(0, cy - half), y1 = Math.min(img.height, cy + half + 1)
  const px: number[][] = []
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = 3 * (y * img.width + x)
      px.push([lab[i], lab[i + 1], lab[i + 2]])
    }
  }
  const mean = [0, 1, 2].map((c) => px.reduce((s, p) => s + p[c], 0) / px.length)
  return { mean, px }
}

export function applyTrichion(fd: FaceData, img: RgbaImage, lab: Uint8Array): number[] | null {
  const fore = fd.top_center_forehead
  const glab = fd.glabella
  let d = [fore[0] - glab[0], fore[1] - glab[1]]
  const dn = Math.hypot(d[0], d[1])
  d = [d[0] / dn, d[1] / dn]
  const start = [fore[0], fore[1]]

  const refPos = [glab[0] + 0.4 * (fore[0] - glab[0]), glab[1] + 0.4 * (fore[1] - glab[1])]
  const window = Math.abs(fd.left_zygomatic[0] - fd.right_zygomatic[0]) * 0.03
  // 3-D distance, as in Python (fore/glab carry z)
  const maxDist = 1.5 * Math.hypot(fore[0] - glab[0], fore[1] - glab[1], fore[2] - glab[2])

  // Skin reference: mean colour and the spread of per-pixel distances to it.
  const ref = windowLab(lab, img, window, refPos)
  let refMean = ref.mean
  const dists = ref.px.map((p) => Math.hypot(p[0] - refMean[0], p[1] - refMean[1], p[2] - refMean[2]))
  const dm = dists.reduce((s, v) => s + v, 0) / dists.length
  const spread = Math.sqrt(dists.reduce((s, v) => s + (v - dm) ** 2, 0) / dists.length)
  const threshold = K * Math.max(spread, FLOOR)

  const minRun = Math.max(3, Math.floor(window / 2))
  const alpha = 0.15
  let pos = [...start]
  let run = 0
  let runStart: number[] | null = null
  while (pos[0] >= 0 && pos[0] < img.width && pos[1] >= 0 && pos[1] < img.height &&
         Math.hypot(pos[0] - start[0], pos[1] - start[1]) < maxDist) {
    const { mean } = windowLab(lab, img, window, pos)
    const dist = Math.hypot(mean[0] - refMean[0], mean[1] - refMean[1], mean[2] - refMean[2])
    if (dist <= threshold) {
      run = 0
      refMean = refMean.map((v, i) => alpha * mean[i] + (1 - alpha) * v)
    } else {
      if (run === 0) runStart = [...pos]
      run += 1
      if (run >= minRun) {
        fd.top_center_forehead = [runStart![0], runStart![1], fore[2]]
        return runStart
      }
    }
    pos = [pos[0] + d[0], pos[1] + d[1]]
  }
  return null                                            // fall back to MediaPipe's point
}
