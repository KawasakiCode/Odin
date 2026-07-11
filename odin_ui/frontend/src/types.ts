export interface RatioItem {
  key: string
  label: string
  value: number | null
  ideal: string | null
  landmarks: number[]
}

export interface ColorItem {
  label: string
  hex: string | null
}

export interface RatioBar {
  pos: number                 // 0..1 marker position on the ideal gradient
  status: 'good' | 'warn' | 'bad'
}

export interface ContribItem {
  key: string
  label: string
  value: number | null
  ideal: string | null
  contribution: number
  landmarks: number[]
  bar: RatioBar | null
  lines: number[][][]          // segments; each is a list of [x,y] points
}

export interface Averageness {
  z: number
  category: string
  percentile: number
}

export interface CnnScores {
  alexnet: number | null
  resnet18: number | null
  resnext50: number | null
}

export interface AnalyzeResult {
  width: number
  height: number
  sex: string
  score: number
  score_raw: number
  cnn_scores: CnnScores | null
  boosted: boolean
  base: number
  landmarks: [number, number][]
  trichion: [number, number] | null
  ratios: RatioItem[]
  appearance: RatioItem[]
  contribs: ContribItem[]
  averageness: Averageness | null
  colors: ColorItem[]
}
