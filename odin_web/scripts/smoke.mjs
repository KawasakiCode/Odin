// End-to-end check of the production build: serve dist/, upload a photo in
// headless Chrome, click Analyze, print the score and save a screenshot.
//   npm run build && node scripts/smoke.mjs path/to/photo.jpg [male|female] [out.png]
//   npm run build && node scripts/smoke.mjs --sample N [out.png]   (click the Nth demo face)
import puppeteer from 'puppeteer-core'
import { preview } from 'vite'

const args = process.argv.slice(2)
const sampleIdx = args[0] === '--sample' ? Number(args[1]) : null
const [photo, sex = 'male', shot = 'smoke.png'] = sampleIdx === null ? args : [null, null, args[2] ?? 'smoke.png']
if (!photo && sampleIdx === null) throw new Error('usage: node scripts/smoke.mjs photo.jpg [male|female] [out.png]')
const CHROME = process.env.CHROME ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe'

const server = await preview({ preview: { port: 5198 }, logLevel: 'error' })
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: process.env.CI ? ['--no-sandbox'] : [],   // GitHub's Linux runners
})
try {
  const page = await browser.newPage()
  await page.setViewport({ width: 1280, height: 900 })
  page.on('pageerror', (e) => console.error('page error:', e.message))
  await page.goto('http://localhost:5198/')
  const t0 = Date.now()
  if (sampleIdx === null) {
    const input = await page.$('input[type=file]')
    await input.uploadFile(photo)
    await page.evaluate((s) => [...document.querySelectorAll('.sex-toggle button')]
      .find((b) => b.textContent === s).click(), sex)
    await page.click('.analyze-btn')
  } else {
    await page.evaluate((i) => document.querySelectorAll('.sample')[i].click(), sampleIdx)
  }
  await page.waitForSelector('.score-value, .error', { timeout: 120_000 })
  const score = await page.$eval('.score-value, .error', (el) => el.textContent)
  const label = await page.$eval('.score-label, .error', (el) => el.textContent)
  console.log(`${label}: ${score}  (${Date.now() - t0} ms incl. first model load)`)
  const isError = await page.$('.error')
  if (sampleIdx !== null && (isError || !/^\d+(\.\d+)?\/10$/.test(score))) {
    console.error('FAIL: the sample did not produce a score')
    process.exitCode = 1
  }
  await page.screenshot({ path: shot, fullPage: true })
  console.log(`screenshot: ${shot}`)
} finally {
  await browser.close()
  server.httpServer.close()
}
