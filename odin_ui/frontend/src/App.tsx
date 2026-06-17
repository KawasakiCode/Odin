import { useEffect, useRef, useState } from 'react'
import { analyze } from './api'
import type { AnalyzeResult } from './types'

type Sex = 'male' | 'female'

function fmt(value: number | null): string {
  return value === null ? '—' : value.toFixed(3)
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [sex, setSex] = useState<Sex>('male')
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showLandmarks, setShowLandmarks] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0]
    if (!picked) return
    setFile(picked)
    setResult(null)
    setError(null)
    setImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(picked)
    })
  }

  async function onAnalyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await analyze(file, sex))
    } catch (err) {
      setError((err as Error).message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  // Draw the uploaded image to the canvas, then overlay the landmark dots.
  // The canvas is sized to the image's natural pixels so the backend's
  // pixel-space landmark coords map 1:1; CSS scales the canvas to fit.
  useEffect(() => {
    if (!imageUrl) return
    const canvas = canvasRef.current
    if (!canvas) return
    const img = new Image()
    img.onload = () => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0)
      if (result && showLandmarks) {
        const radius = Math.max(1, Math.round(canvas.width / 260))
        ctx.fillStyle = 'rgba(0, 230, 150, 0.85)'
        for (const [x, y] of result.landmarks) {
          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }
    img.src = imageUrl
  }, [imageUrl, result, showLandmarks])

  return (
    <div className="app">
      <header>
        <h1>Odin</h1>
        <span className="subtitle">Facial ratio &amp; attractiveness analysis</span>
      </header>

      <div className="controls">
        <label className="file-btn">
          {file ? 'Change photo' : 'Upload photo'}
          <input type="file" accept="image/*" onChange={onPick} hidden />
        </label>

        <div className="sex-toggle">
          {(['male', 'female'] as Sex[]).map((s) => (
            <button
              key={s}
              className={sex === s ? 'active' : ''}
              onClick={() => setSex(s)}
              type="button"
            >
              {s}
            </button>
          ))}
        </div>

        <button
          className="analyze-btn"
          onClick={onAnalyze}
          disabled={!file || loading}
          type="button"
        >
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={showLandmarks}
            onChange={(e) => setShowLandmarks(e.target.checked)}
          />
          Show landmarks
        </label>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="layout">
        <div className="canvas-pane">
          {imageUrl ? (
            <canvas ref={canvasRef} />
          ) : (
            <div className="placeholder">Upload a face photo to begin</div>
          )}
        </div>

        <aside className="results">
          {result ? (
            <>
              <div className="score-card">
                <div className="score-value">{result.score.toFixed(2)}</div>
                <div className="score-label">/ 10 · {result.sex}</div>
                {result.boosted && (
                  <div className="score-note">raw {result.score_raw.toFixed(2)} · male boost applied</div>
                )}
              </div>

              {result.colors.some((c) => c.hex) && (
                <div className="swatches">
                  {result.colors.map((c) => (
                    <div className="swatch" key={c.label}>
                      <span className="chip" style={{ background: c.hex ?? 'transparent' }} />
                      <span>{c.label}</span>
                      <span className="hex">{c.hex ?? '—'}</span>
                    </div>
                  ))}
                </div>
              )}

              <h3>Ratios</h3>
              <table>
                <tbody>
                  {result.ratios.map((r) => (
                    <tr key={r.key}>
                      <td>{r.label}</td>
                      <td className="num">{fmt(r.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h3>Appearance</h3>
              <table>
                <tbody>
                  {result.appearance.map((r) => (
                    <tr key={r.key}>
                      <td>{r.label}</td>
                      <td className="num">{fmt(r.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <div className="placeholder small">Results will appear here</div>
          )}
        </aside>
      </div>
    </div>
  )
}
