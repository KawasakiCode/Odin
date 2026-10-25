import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative base so the build works from any GitHub Pages path
// (https://<user>.github.io/<repo>/). parity.html is dev-only: the build
// only bundles index.html.
export default defineConfig({
  base: './',
  plugins: [react()],
})
