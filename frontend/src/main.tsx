/**
 * main.tsx — Entry point aplikasi React.
 *
 * Merender komponen root <App /> ke dalam elemen #root di index.html.
 * StrictMode mengaktifkan pemeriksaan tambahan saat development
 * (misalnya mendeteksi side effect yang tidak aman).
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
