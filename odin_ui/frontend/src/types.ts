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

export interface ContribItem {
  key: string
  label: string
  value: number | null
  contribution: number
  landmarks: number[]
}

export interface AnalyzeResult {
  width: number
  height: number
  sex: string
  score: number
  score_raw: number
  boosted: boolean
  base: number
  landmarks: [number, number][]
  trichion: [number, number] | null
  ratios: RatioItem[]
  appearance: RatioItem[]
  contribs: ContribItem[]
  colors: ColorItem[]
}
