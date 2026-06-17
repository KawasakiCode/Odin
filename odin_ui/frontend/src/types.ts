export interface RatioItem {
  key: string
  label: string
  value: number | null
}

export interface ColorItem {
  label: string
  hex: string | null
}

export interface AnalyzeResult {
  width: number
  height: number
  sex: string
  score: number
  score_raw: number
  boosted: boolean
  landmarks: [number, number][]
  ratios: RatioItem[]
  appearance: RatioItem[]
  colors: ColorItem[]
}
