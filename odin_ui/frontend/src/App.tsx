import { useEffect, useRef, useState } from 'react'
import { analyze } from './api'
import type { AnalyzeResult, Averageness } from './types'

type Sex = 'male' | 'female'

const fmtScore = (v: number | null) => (v == null ? '—' : v.toFixed(1))

// Bell curve of shape typicality (averageness z-score) with a marker at the face.
function BellCurve({ a }: { a: Averageness }) {
  const W = 240, H = 66, pad = 8, base = H - 10, peak = base - 8
  const zToX = (z: number) => pad + ((z + 3.2) / 6.4) * (W - 2 * pad)
  const yAt = (z: number) => base - Math.exp(-(z * z) / 2) * peak
  const pts: string[] = []
  for (let z = -3.2; z <= 3.201; z += 0.2) pts.push(`${zToX(z).toFixed(1)},${yAt(z).toFixed(1)}`)
  const zc = Math.max(-3.2, Math.min(3.2, a.z))
  const mx = zToX(zc)
  return (
    <div className="bell-card">
      <div className="bell-head">
        <span className="bell-title">Shape typicality</span>
        <span className="bell-cat">{a.category}</span>
      </div>
      <svg className="bell-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {[-2, -1, 1, 2].map((z) => (
          <line key={z} x1={zToX(z)} y1={12} x2={zToX(z)} y2={base}
                stroke="var(--border)" strokeWidth="1" strokeDasharray="2 3" />
        ))}
        <line x1={pad} y1={base} x2={W - pad} y2={base} stroke="var(--border)" strokeWidth="1" />
        <path d={'M' + pts.join(' L')} fill="none" stroke="var(--muted)" strokeWidth="1.5" />
        <line x1={mx} y1={yAt(zc)} x2={mx} y2={base} stroke="var(--accent)" strokeWidth="2" />
        <circle cx={mx} cy={yAt(zc)} r="3.5" fill="var(--accent)" />
      </svg>
      <div className="bell-foot">more distinct than {a.percentile}% of faces</div>
      <div className="bell-note">
        How usual vs. distinctive your shape is — not an attractiveness rating.
      </div>
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [sex, setSex] = useState<Sex>('male')
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showLandmarks, setShowLandmarks] = useState(true)
  const [hovered, setHovered] = useState<string | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgElRef = useRef<HTMLImageElement | null>(null)
  const loadedUrlRef = useRef<string | null>(null)

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0]
    if (!picked) return
    setFile(picked)
    setResult(null)
    setError(null)
    setHovered(null)
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

  // Draw the photo + landmark overlay. The image element is cached so that
  // hovering a ratio (which only changes which dots are drawn) doesn't reload
  // the image and flicker.
  useEffect(() => {
    if (!imageUrl) return
    const canvas = canvasRef.current
    if (!canvas) return

    const paint = (img: HTMLImageElement) => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0)
      if (!result || !showLandmarks) return

      const item = hovered ? result.contribs.find((r) => r.key === hovered) : undefined

      // Hovering a geometric ratio -> draw the ACTUAL lines that were measured
      // (dark underlay for contrast, then the bright line, then endpoint dots).
      if (item && item.lines.length) {
        const lw = Math.max(1, Math.round(canvas.width / 550))
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        // Thin red line, with a faint dark underlay so it reads on light skin.
        for (const pass of [
          { w: lw + Math.max(1.5, lw), s: 'rgba(0, 0, 0, 0.4)' },
          { w: lw, s: 'rgba(229, 57, 53, 0.95)' },
        ]) {
          ctx.lineWidth = pass.w
          ctx.strokeStyle = pass.s
          for (const seg of item.lines) {
            ctx.beginPath()
            seg.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])))
            ctx.stroke()
          }
        }
        return
      }

      // Default (or a feature with no geometry): faint union of every ratio's
      // landmarks, with the corrected trichion in place of raw landmark 10.
      const used = new Set<number>()
      result.contribs.forEach((r) => r.landmarks.forEach((i) => used.add(i)))
      const radius = Math.max(2, Math.round(canvas.width / 220))
      ctx.lineWidth = Math.max(1, radius * 0.5)
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)'
      ctx.fillStyle = 'rgba(0, 230, 150, 0.95)'
      for (const i of used) {
        if (i === 10) continue
        const p = result.landmarks[i]
        if (!p) continue
        ctx.beginPath()
        ctx.arc(p[0], p[1], radius, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
      }
      if (result.trichion && used.has(10)) {
        const [tx, ty] = result.trichion
        ctx.beginPath()
        ctx.arc(tx, ty, radius, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
      }
    }

    const cached = imgElRef.current
    if (cached && loadedUrlRef.current === imageUrl) {
      paint(cached)
    } else {
      const img = new Image()
      img.onload = () => {
        imgElRef.current = img
        loadedUrlRef.current = imageUrl
        paint(img)
      }
      img.src = imageUrl
    }
  }, [imageUrl, result, showLandmarks, hovered])

  return (
    <div className="app">
      <header>
        <h1>Odin</h1>
        <span className="subtitle">Facial geometry &amp; ratio analysis</span>
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
                <div className="score-value">
                  {result.score}<span className="score-max">/10</span>
                </div>
                <div className="score-label">{result.sex} · Odin</div>
                {result.cnn_scores && (
                  <>
                    <div className="cnn-row">
                      <div className="cnn-chip">
                        <span className="cnn-name">AlexNet</span>
                        <span className="cnn-val">{fmtScore(result.cnn_scores.alexnet)}</span>
                      </div>
                      <div className="cnn-chip">
                        <span className="cnn-name">ResNet-18</span>
                        <span className="cnn-val">{fmtScore(result.cnn_scores.resnet18)}</span>
                      </div>
                      <div className="cnn-chip">
                        <span className="cnn-name">ResNeXt-50</span>
                        <span className="cnn-val">{fmtScore(result.cnn_scores.resnext50)}</span>
                      </div>
                    </div>
                    <div className="cnn-note">
                      SCUT-FBP5500 benchmark CNNs (~0.81 R²) · rescaled to /10
                    </div>
                  </>
                )}
              </div>

              {result.averageness && <BellCurve a={result.averageness} />}

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

              <div className="metrics">
                <h3>What drove the result{' '}
                  <span className="hint">green = ideal · hover a ratio to see it measured on the photo</span>
                </h3>
                <div className="ratio-list">
                  {result.contribs.map((r) => {
                    const pos = r.contribution >= 0
                    return (
                      <div
                        key={r.key}
                        className={'ratio-row' + (hovered === r.key ? ' hl' : '')}
                        onMouseEnter={() => setHovered(r.key)}
                        onMouseLeave={() => setHovered((h) => (h === r.key ? null : h))}
                        title={r.ideal ? `Ideal (${result.sex}): ${r.ideal}` : undefined}
                      >
                        <div className="rr-label">
                          <span className="rr-name">{r.label}</span>
                          {r.value !== null && (
                            <span className="rr-badge">{r.value}</span>
                          )}
                        </div>
                        <div className="rr-bar">
                          {r.bar && (
                            <div className="grad-bar">
                              <span
                                className="grad-marker"
                                data-status={r.bar.status}
                                style={{ left: `${(r.bar.pos * 100).toFixed(1)}%` }}
                              />
                            </div>
                          )}
                        </div>
                        <span
                          className="rr-impact"
                          style={{ color: pos ? 'var(--accent)' : '#ff6b6b' }}
                        >
                          {pos ? '+' : ''}{r.contribution.toFixed(2)}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            </>
          ) : (
            <div className="placeholder small">Results will appear here</div>
          )}
        </aside>
      </div>
    </div>
  )
}
