/**
 * Everything after landmark detection: pixel-space landmarks + image ->
 * features -> score + contributions. Mirrors Odin/main.py (build_features,
 * add_shape_features) and the analyze() route of odin_ui/backend/app.py.
 * DOM-free, so the parity test runs it in Node against the Python output.
 */
import { appearanceFeatures, extractRegions, type Regions } from './appearance'
import { extractFaceData, type FaceData, type Vec } from './faceData'
import { toGray, toLab, type RgbaImage } from './imageOps'
import { contributions, predict, type OdinModel } from './model'
import * as R from './ratios'
import { shapeFeatures } from './shape'
import { applyTrichion } from './trichion'

export function buildFeatures(fd: FaceData, appearance: Record<string, number>): Record<string, number> {
  const jaw = R.frontalJawContourAngle(fd)
  const thirds = R.horizontalThirds(fd)
  const fifths = R.facialFifths(fd)
  const lower = R.lowerThirdSplit(fd)
  const ear = R.eyeAspectRatio(fd)
  return {
    symmetry: R.symmetryScore(fd),
    width_ratio_46: R.widthRatio46(fd),
    height_ratio_36: R.heightRatio36(fd),
    canthal_average: R.canthalTiltFinal(fd),
    fwhr: R.fwhr(fd),
    jaw_contour_deviation: jaw.deviation,
    jaw_contour_jaw_slope: jaw.jaw_slope,
    jaw_contour_canthus_alare_slope: jaw.canthus_alare_slope,
    upper_third: thirds.upper_perc,
    middle_third: thirds.middle_perc,
    lower_third: thirds.lower_perc,
    bizygomatic_bigonial_ratio: R.bizygomaticBigonialRatio(fd),
    facial_fifths: fifths.fifths_ratio,
    inter_eye_ratio: fifths.inter_eye_ratio,
    orbitonasal_ratio: R.orbitonasalRatio(fd),
    nasofacial_proportion: R.nasofacialProportion(fd),
    naso_oral_ratio: R.nasoOralRatio(fd),
    face_golden_ratio: R.faceGoldenRatio(fd),
    face_height_bigonial_width: R.faceHeightBigonialWidth(fd),
    lip_vermilion_ratio: R.lipVermilionRatio(fd),
    lower_third_upper_split: lower.upper_pct,
    lower_third_lower_split: lower.lower_pct,
    stomion_canthus_ratio: R.stomionCanthusRatio(fd),
    ear_avg: ear.average,
    ear_asymmetry: ear.asymmetry,
    ...appearance,
  }
}

/** Normalised MediaPipe landmarks -> pixel space (x*w, y*h, z*w), as in training. */
export function toPixelSpace(normalised: Vec[], width: number, height: number): Vec[] {
  return normalised.map(([x, y, z]) => [x * width, y * height, z * width])
}

export interface Analysis {
  px: Vec[]
  faceData: FaceData
  trichion: number[] | null
  regions: Regions
  features: Record<string, number>
  score: number
  base: number
  contribs: Record<string, number>
}

export function analyzeFace(img: RgbaImage, px: Vec[], model: OdinModel): Analysis {
  const cache = { img, lab: toLab(img), gray: toGray(img) }
  const faceData = extractFaceData(px)
  // Detected hairline replaces MediaPipe's forehead point before any ratio.
  const trichion = applyTrichion(faceData, img, cache.lab)
  const regions = extractRegions(px)
  const features = {
    ...buildFeatures(faceData, appearanceFeatures(cache, px, regions)),
    ...shapeFeatures(px, model.shape),
  }
  const x = model.featureNames.map((k) => features[k] ?? NaN)
  const { base, phi } = contributions(model, x)
  const contribs: Record<string, number> = {}
  model.featureNames.forEach((k, i) => { contribs[k] = phi[i] })
  return { px, faceData, trichion, regions, features, score: predict(model, x), base, contribs }
}

/** Male-only presentation stretch (Odin/main.py male_boost); unused by the UI. */
export function maleBoost(raw: number, model: OdinModel, anchorScore = 6.28, anchorAmount = 1.5): number {
  const mean = model.xgbPredMean, cap = model.xgbPredMax
  if (mean == null || cap == null || raw <= mean) return raw
  return Math.min(raw + (anchorAmount / (anchorScore - mean)) * (raw - mean), cap)
}
