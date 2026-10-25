/**
 * Photo-quality checks: head pose and expression vs. what the model was trained
 * on. SCUT-FBP5500's photos are frontal and mostly neutral, and the ratios and
 * Procrustes shape assume that - a turned or tilted head changes the measured
 * proportions (Procrustes removes in-plane rotation, but roll still shifts every
 * vertical ratio and the canthal tilt). Thresholds sit just outside the range
 * of SCUT's own photos (MediaPipe on 400 random training faces):
 *
 *            p1      p5     p50     p95     p99
 *   yaw    -13.2   -7.4    -0.4     6.6    11.6   deg
 *   pitch   -7.9   -2.6     5.6    15.5    19.5   deg
 *   roll    -7.8   -4.7     0.2     6.0     9.3   deg
 *   smile    0.00   0.00    0.05    0.93    0.97  (mouthSmile blendshape)
 *   jawOpen  0.00   0.00    0.01    0.17    0.32
 *
 * Smiling is common in SCUT itself, so it only gets a soft note.
 */

export interface Pose { yaw: number; pitch: number; roll: number }   // degrees
export interface Expression { smile: number; jawOpen: number }      // 0..1 blendshapes
export interface PhotoWarning { level: 'warn' | 'info'; text: string }

const YAW_MAX = 10
const ROLL_MAX = 8
const PITCH_MIN = -10
const PITCH_MAX = 22
const JAW_OPEN_MAX = 0.35
const SMILE_NOTE = 0.6

const DEG = 180 / Math.PI

/**
 * Euler angles from MediaPipe's facial transformation matrix (row-major 4x4,
 * as Python's np.array(matrix) gives it). Same convention as the analysis that
 * produced the thresholds above.
 */
export function poseFromMatrix(m: ArrayLike<number>): Pose {
  // The JS Matrix doesn't document its layout: the translation (a large z) is
  // in the last column, so whichever of elements 12-14 or 3/7/11 holds it tells
  // row- from column-major.
  const colMajor = Math.abs(m[12]) + Math.abs(m[13]) + Math.abs(m[14]) >
    Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11])
  const r = (i: number, j: number) => (colMajor ? m[4 * j + i] : m[4 * i + j])
  return {
    yaw: Math.atan2(-r(2, 0), Math.hypot(r(0, 0), r(1, 0))) * DEG,
    pitch: Math.atan2(r(2, 1), r(2, 2)) * DEG,
    roll: Math.atan2(r(1, 0), r(0, 0)) * DEG,
  }
}

const deg = (v: number) => `${Math.round(Math.abs(v))}°`

export function photoWarnings(pose: Pose | null, expr: Expression | null): PhotoWarning[] {
  const out: PhotoWarning[] = []
  const unreliable = 'so the proportions and the score are less reliable. A straight-on photo works best.'
  if (pose) {
    if (Math.abs(pose.yaw) > YAW_MAX) {
      out.push({ level: 'warn', text: `Head turned ${deg(pose.yaw)} to the side — the model was trained on front-facing photos (almost all within 7°), ${unreliable}` })
    }
    if (Math.abs(pose.roll) > ROLL_MAX) {
      out.push({ level: 'warn', text: `Head tilted ${deg(pose.roll)} sideways — this skews the eye tilt and every vertical ratio, ${unreliable}` })
    }
    if (pose.pitch < PITCH_MIN || pose.pitch > PITCH_MAX) {
      out.push({ level: 'warn', text: `Head tilted strongly up or down (${deg(pose.pitch - 5.6)} from the typical training photo), ${unreliable}` })
    }
  }
  if (expr) {
    if (expr.jawOpen > JAW_OPEN_MAX) {
      out.push({ level: 'warn', text: `Mouth open — lip, chin and lower-face measurements assume a closed mouth, ${unreliable.replace('A straight-on photo', 'A neutral expression')}` })
    } else if (expr.smile > SMILE_NOTE) {
      out.push({ level: 'info', text: 'Smiling changes the lip and jaw measurements a little; a neutral expression gives the most comparable result.' })
    }
  }
  return out
}
