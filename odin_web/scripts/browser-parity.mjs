// Run parity.html in headless Chrome against the Vite dev server and print the
// landmark/score drift between browser MediaPipe and Python MediaPipe.
//   npm run parity:browser            (CHROME=path/to/chrome to override)
import puppeteer from 'puppeteer-core'
import { createServer } from 'vite'

const CHROME = process.env.CHROME ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe'

const server = await createServer({ server: { port: 5199 }, logLevel: 'error' })
await server.listen()
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true })
try {
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))
  await page.goto('http://localhost:5199/parity.html')
  await page.waitForFunction('window.__parity', { timeout: 600_000 })
  const res = await page.evaluate('window.__parity')
  if (res.error) throw new Error(res.error)
  for (const r of res.rows) {
    console.log(`${r.id.padEnd(16)} landmarks Δ mean ${r.meanPx.toFixed(3)} max ${r.maxPx.toFixed(3)} px` +
      `   score js ${r.jsScore.toFixed(3)} py ${r.pyScore.toFixed(3)} Δ ${(r.jsScore - r.pyScore).toFixed(4)}`)
  }
  console.log(res.summary)
} finally {
  await browser.close()
  await server.close()
}
