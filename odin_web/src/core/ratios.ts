/**
 * Port of Odin/Face_analysis/Ratios/calculate_ratios.py + utils.py.
 * Note which ratios use full 3-D distances (symmetry / width_ratio_46 divide
 * by the 3-D bizygomatic width) vs. 2-D or single-axis ones - kept as in Python.
 */
import type { FaceData, Vec } from './faceData'

const DEG = 180 / Math.PI
const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0))
const sub = (a: Vec, b: Vec) => a.map((x, i) => x - b[i])
const d2 = (a: Vec, b: Vec) => Math.hypot(a[0] - b[0], a[1] - b[1])
const d3 = (a: Vec, b: Vec) => norm(sub(a, b))
const stomion = (fd: FaceData) =>
  [0, 1].map((i) => (fd.upper_lip_bottom_center[i] + fd.lower_lip_top_center[i]) / 2)

export const SYMMETRIC_PAIR_KEYS: [string, string][] = [
  ['left_eye_outer_corner', 'right_eye_outer_corner'],
  ['left_eye_inner_corner', 'right_eye_inner_corner'],
  ['left_upper_eyelid_center', 'right_upper_eyelid_center'],
  ['left_lower_eyelid_center', 'right_lower_eyelid_center'],
  ['left_eyebrow_upper_outer_point', 'right_eyebrow_upper_outer_point'],
  ['left_eyebrow_upper_inner_point', 'right_eyebrow_upper_inner_point'],
  ['left_eyebrow_lower_outer_point', 'right_eyebrow_lower_outer_point'],
  ['left_eyebrow_lower_inner_point', 'right_eyebrow_lower_inner_point'],
  ['left_eyebrow_peak_from_eye', 'right_eyebrow_peak_from_eye'],
  ['left_eyebrow_peak_from_forehead', 'right_eyebrow_peak_from_forehead'],
  ['left_alare_tip', 'right_alare_tip'],
  ['lip_left_outer', 'lip_right_outer'],
  ['left_zygomatic', 'right_zygomatic'],
  ['left_jaw_angle1', 'right_jaw_angle1'],
  ['left_jaw_angle2', 'right_jaw_angle2'],
  ['left_cheek_apex', 'right_cheek_apex'],
  ['left_cheek_hollow', 'right_cheek_hollow'],
]

export const SYMMETRY_MIDLINE = ['top_center_forehead', 'glabella', 'top_of_nose_bridge',
  'nose_tip', 'base_of_nose', 'upper_lip_top_center', 'chin']

/** Unsigned angle between two segments, folded to [0, 90] degrees. */
function acuteAngleBetween(p1: Vec, p2: Vec, q1: Vec, q2: Vec): number {
  // Python passes [:2] slices: these are 2-D vectors.
  const v1 = [p2[0] - p1[0], p2[1] - p1[1]], v2 = [q2[0] - q1[0], q2[1] - q1[1]]
  const n1 = norm(v1), n2 = norm(v2)
  if (n1 === 0 || n2 === 0) return 0
  const c = Math.abs(v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
  return Math.acos(Math.min(1, Math.max(-1, c))) * DEG
}

const descentBelowHorizontal = (p1: Vec, p2: Vec) =>
  Math.atan2(Math.abs(p2[1] - p1[1]), Math.abs(p2[0] - p1[0])) * DEG

export function symmetryScore(fd: FaceData): number {
  const midX = SYMMETRY_MIDLINE.reduce((s, k) => s + fd[k][0], 0) / SYMMETRY_MIDLINE.length
  let sq = 0
  for (const [l, r] of SYMMETRIC_PAIR_KEYS) {
    const dx = 2 * midX - fd[l][0] - fd[r][0]
    const dy = fd[l][1] - fd[r][1]
    sq += dx * dx + dy * dy
  }
  return Math.sqrt(sq) / d3(fd.left_zygomatic, fd.right_zygomatic)
}

export const widthRatio46 = (fd: FaceData) =>
  d2(fd.left_pupil_center, fd.right_pupil_center) / d3(fd.left_zygomatic, fd.right_zygomatic)

export function heightRatio36(fd: FaceData): number {
  const pupils = (fd.left_pupil_center[1] + fd.right_pupil_center[1]) / 2
  return Math.abs(pupils - stomion(fd)[1]) / Math.abs(fd.top_center_forehead[1] - fd.chin[1])
}

const canthalTilt = (inner: Vec, outer: Vec) =>
  Math.atan2(-(outer[1] - inner[1]), Math.abs(outer[0] - inner[0])) * DEG

export const canthalTiltFinal = (fd: FaceData) =>
  (canthalTilt(fd.left_eye_inner_corner, fd.left_eye_outer_corner) +
    canthalTilt(fd.right_eye_inner_corner, fd.right_eye_outer_corner)) / 2

export const fwhr = (fd: FaceData) =>
  d2(fd.left_zygomatic, fd.right_zygomatic) / d2(fd.eyebrows_bottom, fd.upper_lip_top_center)

export function frontalJawContourAngle(fd: FaceData) {
  const side = (s: 'left' | 'right') => ({
    dev: acuteAngleBetween(fd[`${s}_eye_outer_corner`], fd[`${s}_alare_tip`],
      fd[`${s}_jaw_angle1`], fd.chin),
    jaw: descentBelowHorizontal(fd[`${s}_jaw_angle1`], fd.chin),
    ref: descentBelowHorizontal(fd[`${s}_eye_outer_corner`], fd[`${s}_alare_tip`]),
  })
  const l = side('left'), r = side('right')
  return {
    deviation: (l.dev + r.dev) / 2,
    jaw_slope: (l.jaw + r.jaw) / 2,
    canthus_alare_slope: (l.ref + r.ref) / 2,
  }
}

export function horizontalThirds(fd: FaceData) {
  const upper = Math.abs(fd.top_center_forehead[1] - fd.glabella[1])
  const middle = Math.abs(fd.glabella[1] - fd.base_of_nose[1])
  const lower = Math.abs(fd.base_of_nose[1] - fd.chin[1])
  const total = upper + middle + lower
  return { upper_perc: upper / total, middle_perc: middle / total, lower_perc: lower / total }
}

export const bizygomaticBigonialRatio = (fd: FaceData) =>
  d2(fd.left_zygomatic, fd.right_zygomatic) / d2(fd.left_jaw_angle2, fd.right_jaw_angle2)

export function facialFifths(fd: FaceData) {
  const lw = Math.abs(fd.left_eye_outer_corner[0] - fd.left_eye_inner_corner[0])
  const rw = Math.abs(fd.right_eye_outer_corner[0] - fd.right_eye_inner_corner[0])
  const eye = (lw + rw) / 2
  return {
    fifths_ratio: Math.abs(fd.left_zygomatic[0] - fd.right_zygomatic[0]) / eye,
    inter_eye_ratio: Math.abs(fd.left_eye_inner_corner[0] - fd.right_eye_inner_corner[0]) / eye,
  }
}

export const orbitonasalRatio = (fd: FaceData) =>
  d2(fd.left_alare_tip, fd.right_alare_tip) / d2(fd.left_eye_inner_corner, fd.right_eye_inner_corner)

export const nasofacialProportion = (fd: FaceData) =>
  d2(fd.left_alare_tip, fd.right_alare_tip) / d2(fd.left_zygomatic, fd.right_zygomatic)

export const nasoOralRatio = (fd: FaceData) =>
  d2(fd.lip_left_outer, fd.lip_right_outer) / d2(fd.left_alare_tip, fd.right_alare_tip)

export const faceGoldenRatio = (fd: FaceData) =>
  d2(fd.top_center_forehead, fd.chin) / d2(fd.left_zygomatic, fd.right_zygomatic)

export const faceHeightBigonialWidth = (fd: FaceData) =>
  d2(fd.top_center_forehead, fd.chin) / d2(fd.left_jaw_angle1, fd.right_jaw_angle1)

export const lipVermilionRatio = (fd: FaceData) =>
  d2(fd.lower_lip_top_center, fd.lower_lip_bottom_center) /
  d2(fd.upper_lip_top_center, fd.upper_lip_bottom_center)

export function lowerThirdSplit(fd: FaceData) {
  const st = stomion(fd)[1]
  const upper = Math.abs(fd.base_of_nose[1] - st)
  const lower = Math.abs(st - fd.chin[1])
  return { upper_pct: upper / (upper + lower), lower_pct: lower / (upper + lower) }
}

export function stomionCanthusRatio(fd: FaceData): number {
  const st = stomion(fd)
  const canthus = (d2(st, fd.left_eye_outer_corner) + d2(st, fd.right_eye_outer_corner)) / 2
  return d2(st, fd.chin) / canthus
}

export function eyeAspectRatio(fd: FaceData) {
  const l = d2(fd.left_upper_eyelid_center, fd.left_lower_eyelid_center) /
    d2(fd.left_eye_outer_corner, fd.left_eye_inner_corner)
  const r = d2(fd.right_upper_eyelid_center, fd.right_lower_eyelid_center) /
    d2(fd.right_eye_outer_corner, fd.right_eye_inner_corner)
  return { average: (l + r) / 2, asymmetry: Math.abs(l - r) }
}
