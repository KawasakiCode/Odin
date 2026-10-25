// Copy MediaPipe's WASM runtime into public/ so the site serves it itself
// (no CDN at runtime). Only the SIMD and no-SIMD builds FilesetResolver picks from.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const src = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm')
const dst = join(root, 'public', 'wasm')
mkdirSync(dst, { recursive: true })
for (const f of ['vision_wasm_internal.js', 'vision_wasm_internal.wasm',
                 'vision_wasm_nosimd_internal.js', 'vision_wasm_nosimd_internal.wasm']) {
  copyFileSync(join(src, f), join(dst, f))
}
if (!existsSync(join(root, 'public', 'models', 'model_male.json'))) {
  console.warn('public/models is empty: run `python export_model.py` from the repo venv first')
}
