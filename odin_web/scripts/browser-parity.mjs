// Run parity.html in headless Chrome against the Vite dev server and print the
// landmark/score drift between browser MediaPipe and Python MediaPipe.
//   npm run parity:browser                      (local SCUT set)
//   npm run parity:browser -- fixtures_ci       (committed sample-face set)
// CHROME=path/to/chrome overrides the browser. Exits non-zero past the limits
// below, so CI fails on landmark drift.
import puppeteer from 'puppeteer-core'
import { createServer } from 'vite'

const CHROME = process.env.CHROME ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const SET = process.argv[2] ?? 'fixtures'
// Browser vs. Python MediaPipe is not bit-identical (different builds/CPUs), so
// allow a little drift; a tree split can flip on a sub-pixel change.
const MAX_MEAN_DRIFT_PX = 0.05
const MAX_SCORE_DIFF = 0.1

const server = await createServer({ server: { port: 5199 }, logLevel: 'error' })
await server.listen()
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: process.env.CI ? ['--no-sandbox'] : [],   // GitHub's Linux runners
})
try {
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))
  await page.goto(`http://localhost:5199/parity.html?set=${SET}`)
  await page.waitForFunction('window.__parity', { timeout: 600_000 })
  const res = await page.evaluate('window.__parity')
  if (res.error) throw new Error(res.error)
  for (const r of res.rows) {
    console.log(`${r.id.padEnd(16)} landmarks Δ mean ${r.meanPx.toFixed(3)} max ${r.maxPx.toFixed(3)} px` +
      `   score js ${r.jsScore.toFixed(3)} py ${r.pyScore.toFixed(3)} Δ ${(r.jsScore - r.pyScore).toFixed(4)}`)
  }
  console.log(res.summary)
  if (res.summary.meanLandmarkDriftPx > MAX_MEAN_DRIFT_PX || res.summary.maxAbsScoreDiff > MAX_SCORE_DIFF) {
    console.error(`FAIL: drift above limits (${MAX_MEAN_DRIFT_PX} px mean, ${MAX_SCORE_DIFF} score)`)
    process.exitCode = 1
  }
} finally {
  await browser.close()
  await server.close()
}
