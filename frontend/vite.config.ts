/// <reference types="vitest/config" />
import { fileURLToPath, URL } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The production build is committed to demo/web/static/ and served by
// `uv run python -m demo.web`, so users never need Node.
const OUT_DIR = fileURLToPath(new URL('../demo/web/static', import.meta.url))
// `npm run dev` proxies API calls to a running `python -m demo.web`.
const API_TARGET = 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    outDir: OUT_DIR,
    emptyOutDir: true,
  },
  server: {
    proxy: { '/api': API_TARGET },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    restoreMocks: true,
  },
})
