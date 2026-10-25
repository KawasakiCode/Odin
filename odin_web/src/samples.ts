// Demo faces for visitors without a photo. All are machine-generated, so no
// real person is being rated, and were picked to be frontal with a neutral
// expression (|yaw| < 5°, |roll| < 4°, low smile/jaw-open blendshapes) — the
// pose SCUT's training photos have, so the ratios and shape typicality are
// meaningful. Machine-generated images have no human author and are treated as
// public domain (Wikimedia Commons' PD-algorithm). Resized to <= 640 px.
export interface Sample {
  file: string
  sex: 'male' | 'female'
  source: string
}

const TPDNE = 'https://thispersondoesnotexist.com/'   // StyleGAN2, generated Oct 2026

export const SAMPLES: Sample[] = [
  { file: 'man-1.jpg', sex: 'male', source: TPDNE },
  { file: 'woman-1.jpg', sex: 'female', source: TPDNE },
  { file: 'man-2.jpg', sex: 'male', source: "https://commons.wikimedia.org/wiki/File:William_Clarke_-_Who's_Lila%3F_(AI).png" },
  { file: 'woman-2.jpg', sex: 'female', source: TPDNE },
]
