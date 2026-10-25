/** Port of Odin/Face_analysis/face_data.py: 478 raw points -> named points. */

export type Vec = number[]                    // [x, y, z] in pixel space
export type FaceData = Record<string, Vec>

export const avg = (...ps: Vec[]): Vec =>
  ps[0].map((_, i) => ps.reduce((s, p) => s + p[i], 0) / ps.length)

// name -> MediaPipe index (composite points are added below)
const POINTS: Record<string, number> = {
  left_upper_eyelid_center: 159, left_lower_eyelid_center: 145,
  left_eye_outer_corner: 33, left_eye_inner_corner: 133,
  right_upper_eyelid_center: 386, right_lower_eyelid_center: 374,
  right_eye_outer_corner: 263, right_eye_inner_corner: 362,
  left_pupil_center: 468, right_pupil_center: 473,
  left_eyebrow_upper_outer_point: 70, left_eyebrow_upper_inner_point: 107,
  left_eyebrow_lower_outer_point: 46, left_eyebrow_lower_inner_point: 55,
  left_eyebrow_peak_from_eye: 52, left_eyebrow_peak_from_forehead: 105,
  right_eyebrow_upper_outer_point: 300, right_eyebrow_upper_inner_point: 336,
  right_eyebrow_lower_outer_point: 276, right_eyebrow_lower_inner_point: 285,
  right_eyebrow_peak_from_eye: 282, right_eyebrow_peak_from_forehead: 334,
  base_of_nose: 2, top_of_nose_bridge: 168, nose_tip: 4,
  left_alare_tip: 48, right_alare_tip: 278,
  lip_left_outer: 61, lip_right_outer: 291,
  upper_lip_top_center: 0, upper_lip_bottom_center: 13,
  lower_lip_top_center: 14, lower_lip_bottom_center: 17,
  chin: 152, left_zygomatic: 234, right_zygomatic: 454,
  left_jaw_angle1: 132, left_jaw_angle2: 58,
  right_jaw_angle1: 361, right_jaw_angle2: 288,
  left_jaw_bottom: 172, right_jaw_bottom: 397,
  left_cheek_apex: 50, left_cheek_hollow: 205,
  right_cheek_apex: 280, right_cheek_hollow: 425,
}

export function extractFaceData(lm: Vec[]): FaceData {
  const fd: FaceData = {}
  for (const [name, i] of Object.entries(POINTS)) fd[name] = [...lm[i]]
  // (lm[55] + lm[285] + lm[46] + lm[276]) / 4, summed in the same order
  fd.eyebrows_bottom = lm[55].map((_, i) => (lm[55][i] + lm[285][i] + lm[46][i] + lm[276][i]) / 4)
  fd.glabella = lm[8].map((_, i) => (lm[8][i] + lm[9][i]) / 2)
  // TRICHION_K = 0 in Python: the estimate is landmark 10 itself, later
  // replaced by the detected hairline (trichion.ts).
  fd.top_center_forehead = [...lm[10]]
  return fd
}
