/**
 * In-browser replacement for the FastAPI /analyze call: decode the photo,
 * run MediaPipe's FaceLandmarker (WASM, CPU), then the ported Odin pipeline.
 * Nothing is uploaded - the photo never leaves the page.
 */
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import type { RgbaImage } from './core/imageOps'
import type { OdinModel } from './core/model'
import { analyzeFace, toPixelSpace } from './core/pipeline'
import { present } from './core/present'
import { photoWarnings, poseFromMatrix, type Expression, type Pose } from './core/quality'
import type { AnalyzeResult } from './types'

const BASE = import.meta.env.BASE_URL

let landmarker: Promise<FaceLandmarker> | null = null
const models: Partial<Record<string, Promise<OdinModel>>> = {}

export function getLandmarker(): Promise<FaceLandmarker> {
  landmarker ??= FilesetResolver.forVisionTasks(`${BASE}wasm`).then((files) =>
    FaceLandmarker.createFromOptions(files, {
      // CPU, like the Python pipeline the model was trained with.
      baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate: 'CPU' },
      runningMode: 'IMAGE',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      // head pose + expression for the photo-quality warnings (same landmarks)
      outputFacialTransformationMatrixes: true,
      outputFaceBlendshapes: true,
    }))
  return landmarker
}

export function getModel(sex: string): Promise<OdinModel> {
  models[sex] ??= fetch(`${BASE}models/model_${sex}.json`).then((r) => {
    if (!r.ok) throw new Error(`Could not load the ${sex} model (${r.status})`)
    return r.json()
  })
  return models[sex]!
}

/** Decode to raw RGBA without colour management (EXIF orientation applied, like cv2.imread). */
export async function decode(file: Blob): Promise<RgbaImage> {
  const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  const canvas = new OffscreenCanvas(bmp.width, bmp.height)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height)
  return { width: canvas.width, height: canvas.height, data }
}

export interface Detection {
  landmarks: number[][]          // normalised [x, y, z]
  pose: Pose | null
  expression: Expression | null
}

/** MediaPipe landmarks, head pose and expression for the image, or null if no face. */
export async function detectFace(img: RgbaImage): Promise<Detection | null> {
  const lm = await getLandmarker()
  const res = lm.detect(new ImageData(new Uint8ClampedArray(img.data), img.width, img.height))
  const face = res.faceLandmarks[0]
  if (!face) return null
  const matrix = res.facialTransformationMatrixes?.[0]
  const shapes = res.faceBlendshapes?.[0]?.categories
  const score = (name: string) => shapes?.find((c) => c.categoryName === name)?.score ?? 0
  return {
    landmarks: face.map((p) => [p.x, p.y, p.z]),
    pose: matrix ? poseFromMatrix(matrix.data) : null,
    expression: shapes
      ? { smile: (score('mouthSmileLeft') + score('mouthSmileRight')) / 2, jawOpen: score('jawOpen') }
      : null,
  }
}

export async function detectLandmarks(img: RgbaImage): Promise<number[][] | null> {
  return (await detectFace(img))?.landmarks ?? null
}

export async function analyze(file: File, sex: string): Promise<AnalyzeResult> {
  const [img, model] = await Promise.all([decode(file), getModel(sex)])
  const det = await detectFace(img)
  if (!det) throw new Error('No face detected in this photo.')
  // Let the "Analyzing…" state paint before the synchronous number-crunching.
  await new Promise((r) => setTimeout(r, 0))
  const a = analyzeFace(img, toPixelSpace(det.landmarks, img.width, img.height), model)
  return {
    ...present(a, model, sex as 'male' | 'female', img.width, img.height),
    pose: det.pose,
    warnings: photoWarnings(det.pose, det.expression),
  }
}
