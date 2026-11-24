/**
 * vite.config.ts — Konfigurasi Vite + React + Tailwind CSS v4
 *
 * @tailwindcss/vite adalah plugin resmi Tailwind CSS v4 untuk Vite.
 * Plugin ini menggantikan kebutuhan postcss.config.js — cukup
 * tambahkan di sini dan tulis @import "tailwindcss" di CSS entry.
 */

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [
    tailwindcss(), // Tailwind CSS v4 — proses @theme & utility classes
    react(),       // Fast Refresh + transformasi JSX/TSX
  ],
})
