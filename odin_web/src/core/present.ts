/**
 * Port of the presentation half of odin_ui/backend/app.py: turns an Analysis
 * into the AnalyzeResult JSON the React UI renders (labels, ideal bars,
 * measurement lines, angle arcs, region outlines, averageness percentile).
 */
import type { AnalyzeResult, ContribItem, RatioItem } from '../types'
import type { FaceData } from './faceData'
import type { OdinModel } from './model'
import type { Analysis } from './pipeline'
import { SYMMETRY_MIDLINE } from './ratios'
import type { Regions } from './appearance'

type Sex = 'male' | 'female'

const LABELS: Record<string, string> = {
  symmetry: 'Facial symmetry (lower = better)',
  width_ratio_46: 'Eye spacing / cheekbone (46% ideal)',
  height_ratio_36: 'Midface height (36% ideal)',
  canthal_average: 'Canthal tilt (°)',
  fwhr: 'Facial width-to-height',
  jaw_contour_deviation: 'Jaw contour deviation (°)',
  jaw_contour_jaw_slope: 'Jaw slope (°)',
  jaw_contour_canthus_alare_slope: 'Canthus–alare slope (°)',
  upper_third: 'Upper third %',
  middle_third: 'Middle third %',
  lower_third: 'Lower third %',
  bizygomatic_bigonial_ratio: 'Cheekbone / jaw taper',
  facial_fifths: 'Facial fifths',
  inter_eye_ratio: 'Inter-eye / eye width',
  orbitonasal_ratio: 'Nose width / eye spacing',
  nasofacial_proportion: 'Nose width / face',
  naso_oral_ratio: 'Mouth / nose width',
  face_golden_ratio: 'Face height / width (golden)',
  face_height_bigonial_width: 'Face height / jaw width',
  lip_vermilion_ratio: 'Lower / upper lip',
  lower_third_upper_split: 'Lower third — upper %',
  lower_third_lower_split: 'Lower third — lower %',
  stomion_canthus_ratio: 'Stomion–menton / canthus',
  ear_avg: 'Eye aspect ratio (openness)',
  ear_asymmetry: 'Eye asymmetry',
  skin_texture: 'Skin texture (Laplacian var.)',
  skin_a_std: 'Skin redness unevenness',
  skin_b_std: 'Skin tone unevenness',
  skin_spot_burden: 'Skin blemish / spot level',
  eye_skin_luminance_contrast: 'Eye–skin luminance contrast',
  lip_skin_luminance_contrast: 'Lip–skin luminance contrast',
  lip_skin_redness_contrast: 'Lip–skin redness contrast',
  eye_skin_redness_contrast: 'Eye–skin redness contrast',
  facial_contrast_avg: 'Facial contrast (avg)',
}

const GEOM_KEYS = [
  'symmetry', 'width_ratio_46', 'height_ratio_36', 'canthal_average', 'fwhr',
  'jaw_contour_deviation', 'jaw_contour_jaw_slope', 'jaw_contour_canthus_alare_slope',
  'upper_third', 'middle_third', 'lower_third', 'bizygomatic_bigonial_ratio',
  'facial_fifths', 'inter_eye_ratio', 'orbitonasal_ratio', 'nasofacial_proportion',
  'naso_oral_ratio', 'face_golden_ratio', 'face_height_bigonial_width',
  'lip_vermilion_ratio', 'lower_third_upper_split', 'lower_third_lower_split',
  'stomion_canthus_ratio', 'ear_avg', 'ear_asymmetry',
]
const APPEARANCE_KEYS = [
  'skin_texture', 'skin_a_std', 'skin_b_std', 'skin_spot_burden',
  'eye_skin_luminance_contrast', 'lip_skin_luminance_contrast',
  'lip_skin_redness_contrast', 'eye_skin_redness_contrast', 'facial_contrast_avg',
]

const RATIO_LANDMARKS: Record<string, number[]> = {
  symmetry: [33, 133, 263, 362, 159, 145, 386, 374, 70, 107, 46, 55, 52, 105,
    300, 336, 276, 285, 282, 334, 48, 278, 61, 291, 234, 454, 132,
    361, 58, 288, 50, 280, 205, 425, 10, 8, 9, 168, 4, 2, 0, 152],
  width_ratio_46: [468, 473, 234, 454],
  height_ratio_36: [468, 473, 13, 14, 10, 8, 9, 152],
  canthal_average: [33, 133, 263, 362],
  fwhr: [234, 454, 55, 285, 46, 276, 0],
  jaw_contour_deviation: [33, 48, 132, 152, 263, 278, 361],
  jaw_contour_jaw_slope: [132, 152, 361],
  jaw_contour_canthus_alare_slope: [33, 48, 263, 278],
  upper_third: [10, 8, 9, 2, 152],
  middle_third: [10, 8, 9, 2, 152],
  lower_third: [10, 8, 9, 2, 152],
  bizygomatic_bigonial_ratio: [234, 454, 58, 288],
  facial_fifths: [33, 133, 263, 362, 234, 454],
  inter_eye_ratio: [33, 133, 263, 362],
  orbitonasal_ratio: [48, 278, 133, 362],
  nasofacial_proportion: [48, 278, 234, 454],
  naso_oral_ratio: [61, 291, 48, 278],
  face_golden_ratio: [10, 8, 9, 152, 234, 454],
  face_height_bigonial_width: [10, 8, 9, 152, 132, 361],
  lip_vermilion_ratio: [0, 13, 14, 17],
  lower_third_upper_split: [2, 13, 14, 152],
  lower_third_lower_split: [2, 13, 14, 152],
  stomion_canthus_ratio: [13, 14, 152, 33, 263],
  ear_avg: [159, 145, 33, 133, 386, 374, 263, 362],
  ear_asymmetry: [159, 145, 33, 133, 386, 374, 263, 362],
}

// (male, female) display strings
const IDEALS: Record<string, [string, string]> = {
  symmetry: ['→ 0', '→ 0'],
  width_ratio_46: ['0.46', '0.46'],
  height_ratio_36: ['0.36', '0.36'],
  canthal_average: ['+3 to +5°', '+5 to +8°'],
  fwhr: ['1.90–2.05', '1.75–1.90'],
  jaw_contour_deviation: ['0–15°', '0–15°'],
  jaw_contour_jaw_slope: ['—', '—'],
  jaw_contour_canthus_alare_slope: ['—', '—'],
  upper_third: ['0.310', '0.295'],
  middle_third: ['0.305', '0.324'],
  lower_third: ['0.385', '0.382'],
  bizygomatic_bigonial_ratio: ['1.128', '1.174'],
  facial_fifths: ['5.00', '5.00'],
  inter_eye_ratio: ['0.95–1.15', '0.95–1.15'],
  orbitonasal_ratio: ['1.00', '≤1.00'],
  nasofacial_proportion: ['0.25', '0.25'],
  naso_oral_ratio: ['1.50–1.62', '1.50–1.62'],
  face_golden_ratio: ['1.35', '1.30'],
  face_height_bigonial_width: ['1.566', '1.613'],
  lip_vermilion_ratio: ['1.60', '1.60'],
  lower_third_upper_split: ['0.30', '0.30'],
  lower_third_lower_split: ['0.70', '0.70'],
  stomion_canthus_ratio: ['0.618', '0.618'],
  ear_avg: ['0.20–0.25', '0.30–0.35'],
  ear_asymmetry: ['→ 0', '→ 0'],
}

// (green_lo, green_hi, red_lo, red_hi) per sex [male, female]
type Range = [number, number, number, number]
const both = (r: Range): [Range, Range] => [r, r]
const LOW_BETTER = new Set(['symmetry', 'ear_asymmetry', 'jaw_contour_deviation'])
const IDEAL_RANGES: Record<string, [Range, Range]> = {
  symmetry: both([0, 0.035, 0, 0.09]),
  width_ratio_46: both([0.44, 0.48, 0.40, 0.50]),
  height_ratio_36: both([0.34, 0.38, 0.33, 0.40]),
  canthal_average: [[3, 5, 0, 8], [5, 8, 2, 11]],
  fwhr: [[1.90, 2.05, 1.70, 2.20], [1.75, 1.90, 1.60, 2.05]],
  jaw_contour_deviation: both([0, 15, 0, 25]),
  upper_third: [[0.29, 0.33, 0.26, 0.36], [0.275, 0.315, 0.245, 0.345]],
  middle_third: [[0.285, 0.325, 0.255, 0.355], [0.304, 0.344, 0.274, 0.374]],
  lower_third: [[0.36, 0.41, 0.32, 0.45], [0.36, 0.405, 0.32, 0.44]],
  bizygomatic_bigonial_ratio: [[1.05, 1.20, 1.00, 1.30], [1.10, 1.25, 1.00, 1.35]],
  facial_fifths: both([4.7, 5.3, 4.5, 5.5]),
  inter_eye_ratio: both([0.95, 1.15, 0.85, 1.25]),
  orbitonasal_ratio: [[0.90, 1.10, 0.80, 1.20], [0.85, 1.05, 0.75, 1.15]],
  nasofacial_proportion: both([0.23, 0.27, 0.20, 0.30]),
  naso_oral_ratio: both([1.50, 1.62, 1.30, 1.75]),
  face_golden_ratio: [[1.30, 1.40, 1.25, 1.45], [1.25, 1.35, 1.20, 1.40]],
  face_height_bigonial_width: [[1.48, 1.65, 1.40, 1.72], [1.55, 1.68, 1.45, 1.75]],
  lip_vermilion_ratio: both([1.40, 1.80, 1.20, 2.10]),
  lower_third_upper_split: both([0.25, 0.35, 0.20, 0.40]),
  lower_third_lower_split: both([0.65, 0.75, 0.60, 0.80]),
  stomion_canthus_ratio: both([0.56, 0.68, 0.50, 0.75]),
  ear_avg: [[0.20, 0.25, 0.15, 0.32], [0.30, 0.35, 0.22, 0.42]],
  ear_asymmetry: both([0, 0.03, 0, 0.10]),
}

const SHAPE_LABELS: Record<Sex, Record<string, string>> = {
  female: {
    averageness: 'Shape typicality',
    shape_pls_01: 'Lower-face length',
    shape_pls_02: 'Nose–mouth spacing & lip fullness',
    shape_pls_03: 'Eye spacing & face width',
    shape_pls_04: 'Jaw slope & asymmetry',
    shape_pls_05: 'Chin height & width-to-height',
    shape_pls_06: 'Lower-third balance',
    shape_pls_09: 'Face width & nose proportion',
  },
  male: {
    averageness: 'Shape typicality',
    shape_pls_01: 'Jaw angularity & face width-to-height',
    shape_pls_02: 'Chin–mouth (lower-third) balance',
    shape_pls_03: 'Nose size & midface proportion',
    shape_pls_05: 'Cheekbone-to-jaw width',
    shape_pls_08: 'Face width & jaw contour',
    shape_pls_11: 'Facial asymmetry',
    shape_pls_16: 'Eye spacing & fifths',
  },
}

const REGION_MAP: Record<string, (keyof Regions)[]> = {
  skin_texture: ['left_cheek', 'right_cheek'],
  skin_a_std: ['left_cheek', 'right_cheek'],
  skin_b_std: ['left_cheek', 'right_cheek'],
  skin_spot_burden: ['left_cheek', 'right_cheek'],
  skin_color: ['left_cheek', 'right_cheek'],
  eye_color: ['left_iris', 'right_iris'],
  lips_color: ['lips'],
  eye_skin_luminance_contrast: ['left_iris', 'right_iris', 'left_cheek', 'right_cheek'],
  eye_skin_redness_contrast: ['left_iris', 'right_iris', 'left_cheek', 'right_cheek'],
  lip_skin_luminance_contrast: ['lips', 'left_cheek', 'right_cheek'],
  lip_skin_redness_contrast: ['lips', 'left_cheek', 'right_cheek'],
  facial_contrast_avg: ['lips', 'left_iris', 'right_iris', 'left_cheek', 'right_cheek'],
}

const ANGLE_RATIOS = new Set(['canthal_average', 'jaw_contour_jaw_slope', 'jaw_contour_canthus_alare_slope'])

const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d
const num = (v: number | undefined | null) =>
  v == null || !Number.isFinite(v) ? null : round(v, 4)

function lerpPos(v: number, xs: number[], ys: number[]): number {
  if (v <= xs[0]) return ys[0]
  for (let i = 0; i < xs.length - 1; i++) {
    if (v <= xs[i + 1]) {
      const d = xs[i + 1] - xs[i]
      const t = d ? (v - xs[i]) / d : 0
      return ys[i] + t * (ys[i + 1] - ys[i])
    }
  }
  return ys[ys.length - 1]
}

function bar(key: string, value: number | undefined, sex: Sex) {
  const rng = IDEAL_RANGES[key]
  if (!rng || value == null || !Number.isFinite(value)) return null
  const [glo, ghi, rlo, rhi] = rng[sex === 'male' ? 0 : 1]
  let xs: number[], ys: number[], status: 'good' | 'warn' | 'bad'
  if (LOW_BETTER.has(key)) {
    xs = [0, ghi, rhi, rhi + (rhi - ghi + 1e-6)]
    ys = [0.45, 0.58, 0.85, 0.97]
    status = value <= ghi ? 'good' : value <= rhi ? 'warn' : 'bad'
  } else {
    const pad = (rhi - rlo) * 0.6
    xs = [rlo - pad, rlo, glo, ghi, rhi, rhi + pad]
    ys = [0.03, 0.15, 0.35, 0.65, 0.85, 0.97]
    status = glo <= value && value <= ghi ? 'good' : rlo <= value && value <= rhi ? 'warn' : 'bad'
  }
  return { pos: round(lerpPos(value, xs, ys), 4), status }
}

function ratioLines(key: string, fd: FaceData): number[][][] {
  const P = (k: string) => [round(fd[k][0], 1), round(fd[k][1], 1)]
  const MID = (a: string, b: string) =>
    [round((fd[a][0] + fd[b][0]) / 2, 1), round((fd[a][1] + fd[b][1]) / 2, 1)]
  const st = MID('upper_lip_bottom_center', 'lower_lip_top_center')
  switch (key) {
    case 'symmetry': {
      const mx = round(SYMMETRY_MIDLINE.reduce((s, k) => s + fd[k][0], 0) / SYMMETRY_MIDLINE.length, 1)
      return [[[mx, round(fd.top_center_forehead[1], 1)], [mx, round(fd.chin[1], 1)]]]
    }
    case 'width_ratio_46':
      return [[P('left_pupil_center'), P('right_pupil_center')], [P('left_zygomatic'), P('right_zygomatic')]]
    case 'height_ratio_36': {
      const mid = MID('left_pupil_center', 'right_pupil_center')
      const off = Math.abs(fd.left_zygomatic[0] - fd.right_zygomatic[0]) * 0.12
      const ex = round(mid[0] - off, 1)
      return [[P('top_center_forehead'), P('chin')], [[ex, mid[1]], [ex, st[1]]]]
    }
    case 'canthal_average':
      return [[P('left_eye_inner_corner'), P('left_eye_outer_corner')],
        [P('right_eye_inner_corner'), P('right_eye_outer_corner')]]
    case 'fwhr':
      return [[P('left_zygomatic'), P('right_zygomatic')], [P('eyebrows_bottom'), P('upper_lip_top_center')]]
    case 'jaw_contour_deviation':
      return [[P('left_jaw_angle1'), P('chin')], [P('left_eye_outer_corner'), P('left_alare_tip')],
        [P('right_jaw_angle1'), P('chin')], [P('right_eye_outer_corner'), P('right_alare_tip')]]
    case 'jaw_contour_jaw_slope':
      return [[P('left_jaw_angle1'), P('chin')], [P('right_jaw_angle1'), P('chin')]]
    case 'jaw_contour_canthus_alare_slope':
      return [[P('left_eye_outer_corner'), P('left_alare_tip')], [P('right_eye_outer_corner'), P('right_alare_tip')]]
    case 'upper_third': return [[P('top_center_forehead'), P('glabella')]]
    case 'middle_third': return [[P('glabella'), P('base_of_nose')]]
    case 'lower_third': return [[P('base_of_nose'), P('chin')]]
    case 'bizygomatic_bigonial_ratio':
      return [[P('left_zygomatic'), P('right_zygomatic')], [P('left_jaw_angle2'), P('right_jaw_angle2')]]
    case 'facial_fifths':
      return [[P('left_zygomatic'), P('right_zygomatic')],
        [P('left_eye_inner_corner'), P('left_eye_outer_corner')],
        [P('right_eye_inner_corner'), P('right_eye_outer_corner')]]
    case 'inter_eye_ratio':
      return [[P('left_eye_inner_corner'), P('right_eye_inner_corner')],
        [P('left_eye_inner_corner'), P('left_eye_outer_corner')],
        [P('right_eye_inner_corner'), P('right_eye_outer_corner')]]
    case 'orbitonasal_ratio':
      return [[P('left_alare_tip'), P('right_alare_tip')], [P('left_eye_inner_corner'), P('right_eye_inner_corner')]]
    case 'nasofacial_proportion':
      return [[P('left_alare_tip'), P('right_alare_tip')], [P('left_zygomatic'), P('right_zygomatic')]]
    case 'naso_oral_ratio':
      return [[P('lip_left_outer'), P('lip_right_outer')], [P('left_alare_tip'), P('right_alare_tip')]]
    case 'face_golden_ratio':
      return [[P('top_center_forehead'), P('chin')], [P('left_zygomatic'), P('right_zygomatic')]]
    case 'face_height_bigonial_width':
      return [[P('top_center_forehead'), P('chin')], [P('left_jaw_angle1'), P('right_jaw_angle1')]]
    case 'lip_vermilion_ratio':
      return [[P('upper_lip_top_center'), P('upper_lip_bottom_center')],
        [P('lower_lip_top_center'), P('lower_lip_bottom_center')]]
    case 'lower_third_upper_split':
    case 'lower_third_lower_split':
      return [[P('base_of_nose'), st], [st, P('chin')]]
    case 'stomion_canthus_ratio':
      return [[st, P('chin')], [st, P('left_eye_outer_corner')], [st, P('right_eye_outer_corner')]]
    case 'ear_avg':
    case 'ear_asymmetry':
      return [[P('left_upper_eyelid_center'), P('left_lower_eyelid_center')],
        [P('left_eye_inner_corner'), P('left_eye_outer_corner')],
        [P('right_upper_eyelid_center'), P('right_lower_eyelid_center')],
        [P('right_eye_inner_corner'), P('right_eye_outer_corner')]]
    default:
      return []
  }
}

function ratioAngles(key: string, fd: FaceData, feats: Record<string, number>) {
  if (!ANGLE_RATIOS.has(key)) return []
  const P = (k: string) => [round(fd[k][0], 1), round(fd[k][1], 1)]
  const deg = Number.isFinite(feats[key]) ? round(feats[key], 1) : null
  const spec = (vertex: number[], end: number[]) => ({ vertex, p1: end, p2: [end[0], vertex[1]], deg })
  if (key === 'canthal_average') {
    return [spec(P('left_eye_inner_corner'), P('left_eye_outer_corner')),
      spec(P('right_eye_inner_corner'), P('right_eye_outer_corner'))]
  }
  if (key === 'jaw_contour_jaw_slope') {
    return [spec(P('left_jaw_angle1'), P('chin')), spec(P('right_jaw_angle1'), P('chin'))]
  }
  return [spec(P('left_eye_outer_corner'), P('left_alare_tip')),
    spec(P('right_eye_outer_corner'), P('right_alare_tip'))]
}

const regionPolygons = (key: string, regions: Regions) =>
  (REGION_MAP[key] ?? []).map((n) => regions[n].map(([x, y]) => [x, y]))

/** Acklam's rational approximation of the standard normal inverse CDF. */
function normInv(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1]
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783]
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
  const lo = 0.02425
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  }
  if (p > 1 - lo) return -normInv(1 - p)
  const q = p - 0.5, r = q * q
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
}

function averagenessInfo(avg: number | undefined, q: number[]) {
  if (avg == null || !Number.isFinite(avg) || !q.length) return null
  // np.interp(avg, q, arange(101)), clamped at the ends
  let pct: number
  if (avg <= q[0]) pct = 0
  else if (avg >= q[q.length - 1]) pct = q.length - 1
  else {
    let i = 0
    while (q[i + 1] < avg) i++
    pct = i + (avg - q[i]) / (q[i + 1] - q[i])
  }
  pct = Math.max(0, Math.min(100, pct))
  const category = pct < 30 ? 'Very typical' : pct < 60 ? 'Typical' : pct < 85 ? 'Distinct'
    : pct < 96 ? 'Unique' : 'Very unique'
  const z = normInv(Math.min(0.999, Math.max(0.001, pct / 100)))
  return { z: round(z, 2), category, percentile: Math.round(pct) }
}

const hex = (f: Record<string, number>, prefix: string) => {
  const ch = ['r', 'g', 'b'].map((c) => f[`${prefix}_${c}`])
  if (ch.some((v) => !Number.isFinite(v))) return null
  return '#' + ch.map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0'))
    .join('').toUpperCase()
}

export function present(a: Analysis, model: OdinModel, sex: Sex, width: number, height: number): AnalyzeResult {
  const idx = sex === 'male' ? 0 : 1
  const f = a.features
  const label = (k: string) => SHAPE_LABELS[sex][k] ??
    (k.startsWith('shape_pls_') ? 'Minor shape axis' : LABELS[k] ?? k)

  const items = (keys: string[]): RatioItem[] => keys.map((k) => ({
    key: k,
    label: LABELS[k] ?? k,
    value: num(f[k]),
    ideal: IDEALS[k]?.[idx] ?? null,
    landmarks: RATIO_LANDMARKS[k] ?? [],
  }))

  const channel = new Set(['eye', 'skin', 'lips'].flatMap((p) => ['r', 'g', 'b'].map((c) => `${p}_${c}`)))
  const contribs: ContribItem[] = model.featureNames.filter((k) => !channel.has(k)).map((k) => ({
    key: k,
    label: label(k),
    value: k in IDEALS ? num(f[k]) : null,
    ideal: IDEALS[k]?.[idx] ?? null,
    contribution: round(a.contribs[k], 3),
    landmarks: RATIO_LANDMARKS[k] ?? [],
    bar: bar(k, f[k], sex),
    lines: ratioLines(k, a.faceData),
    angles: ratioAngles(k, a.faceData, f),
    polygons: regionPolygons(k, a.regions),
  }))
  // Colour channels mean nothing alone: group each colour's 3 SHAP values.
  for (const [prefix, lab] of [['eye', 'Eye colour'], ['skin', 'Skin colour'], ['lips', 'Lip colour']]) {
    const total = ['r', 'g', 'b'].reduce((s, c) => s + (a.contribs[`${prefix}_${c}`] ?? 0), 0)
    contribs.push({
      key: `${prefix}_color`, label: lab, value: null, ideal: null,
      contribution: round(total, 3), landmarks: [], bar: null, lines: [], angles: [],
      polygons: regionPolygons(`${prefix}_color`, a.regions),
    })
  }
  contribs.sort((x, y) => Math.abs(y.contribution) - Math.abs(x.contribution))

  // How much of this prediction came from face shape vs. skin/colour: share of
  // the summed |SHAP| (appearance = the colour, texture and contrast features).
  const APPEARANCE = new Set([...APPEARANCE_KEYS, ...channel])
  let shapeAbs = 0, appAbs = 0
  for (const k of model.featureNames) {
    if (APPEARANCE.has(k)) appAbs += Math.abs(a.contribs[k])
    else shapeAbs += Math.abs(a.contribs[k])
  }
  const shapePct = Math.round((100 * shapeAbs) / (shapeAbs + appAbs || 1))

  const fore = a.faceData.top_center_forehead
  return {
    width,
    height,
    sex,
    score: round(a.score, 2),
    score_raw: round(a.score, 2),
    cnn_scores: null,
    boosted: false,
    base: round(a.base, 3),
    landmarks: a.px.map((p) => [round(p[0], 1), round(p[1], 1)] as [number, number]),
    trichion: a.trichion ? [round(a.trichion[0], 1), round(a.trichion[1], 1)] : null,
    forehead: [round(fore[0], 1), round(fore[1], 1)],
    ratios: items(GEOM_KEYS),
    appearance: items(APPEARANCE_KEYS),
    contribs,
    averageness: averagenessInfo(f.averageness, model.averagenessQuantiles),
    colors: [
      { label: 'Lips', hex: hex(f, 'lips') },
      { label: 'Eyes', hex: hex(f, 'eye') },
      { label: 'Skin', hex: hex(f, 'skin') },
    ],
    breakdown: { shape: shapePct, appearance: 100 - shapePct },
  }
}
