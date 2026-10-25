/**
 * Dev-only check of the one stage the Node parity test can't cover: MediaPipe
 * in the browser (WASM) vs. MediaPipe in Python on the exact same pixels.
 * Reports landmark drift and how far it moves the final score. Results are
 * also published on window.__parity for scripts/browser-parity.mjs.
 */
import { decode, detectLandmarks, getModel } from './api'
import { analyzeFace, toPixelSpace } from './core/pipeline'

interface Fixture {
  id: string
  sex: string
  width: number
  height: number
  landmarks: number[][]
  score: number
}

interface Row { id: string; meanPx: number; maxPx: number; jsScore: number; pyScore: number }

async function run() {
  // ?set=fixtures_ci for the committed sample-face set (default: local SCUT set)
  const set = new URLSearchParams(location.search).get('set') ?? 'fixtures'
  const fixtures: Fixture[] = await (await fetch(`/parity/${set}/fixtures.json`)).json()
  const table = document.getElementById('table')!
  table.innerHTML = '<tr><th>fixture</th><th>mean lm Δ px</th><th>max lm Δ px</th>' +
    '<th>JS score</th><th>Py score</th><th>Δ score</th></tr>'
  const rows: Row[] = []
  for (const f of fixtures) {
    // PNG without colour chunks: decodes to exactly the pixels OpenCV saw
    const img = await decode(await (await fetch(`/parity/${set}/${f.id}.png`)).blob())
    const lm = await detectLandmarks(img)
    if (!lm) { table.insertAdjacentHTML('beforeend', `<tr><td>${f.id}</td><td colspan=5>no face</td></tr>`); continue }
    let sum = 0, max = 0
    lm.forEach((p, i) => {
      const d = Math.hypot((p[0] - f.landmarks[i][0]) * f.width, (p[1] - f.landmarks[i][1]) * f.height)
      sum += d
      max = Math.max(max, d)
    })
    const a = analyzeFace(img, toPixelSpace(lm, f.width, f.height), await getModel(f.sex))
    const row = { id: f.id, meanPx: sum / lm.length, maxPx: max, jsScore: a.score, pyScore: f.score }
    rows.push(row)
    table.insertAdjacentHTML('beforeend', `<tr><td>${f.id}</td><td>${row.meanPx.toFixed(3)}</td>` +
      `<td>${row.maxPx.toFixed(3)}</td><td>${row.jsScore.toFixed(3)}</td><td>${row.pyScore.toFixed(3)}</td>` +
      `<td>${(row.jsScore - row.pyScore).toFixed(4)}</td></tr>`)
  }
  const dScore = rows.map((r) => Math.abs(r.jsScore - r.pyScore))
  const summary = {
    fixtures: rows.length,
    meanLandmarkDriftPx: rows.reduce((s, r) => s + r.meanPx, 0) / rows.length,
    maxLandmarkDriftPx: Math.max(...rows.map((r) => r.maxPx)),
    meanAbsScoreDiff: dScore.reduce((s, v) => s + v, 0) / dScore.length,
    maxAbsScoreDiff: Math.max(...dScore),
  }
  document.getElementById('summary')!.textContent = JSON.stringify(summary, null, 2)
  document.getElementById('status')!.textContent = 'done'
  ;(window as unknown as { __parity: unknown }).__parity = { summary, rows }
}

run().catch((e) => {
  document.getElementById('status')!.textContent = `error: ${e}`
  ;(window as unknown as { __parity: unknown }).__parity = { error: String(e) }
})
