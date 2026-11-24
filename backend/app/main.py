"""
main.py — Entry point aplikasi FastAPI.

Menginisialisasi aplikasi, mengatur middleware CORS agar frontend
(Vite dev server) dapat mengakses API, lalu mendaftarkan router API
dengan prefix /api.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import router as api_router

# Inisialisasi aplikasi FastAPI dengan metadata
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="RAG Chatbot API untuk Jaringan Komputer"
)

# Middleware CORS: mengizinkan frontend (localhost Vite) untuk mengakses API.
# allow_origin_regex mencakup semua port localhost agar fleksibel saat development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftarkan semua endpoint API di bawah prefix /api (contoh: /api/chat, /api/health)
app.include_router(api_router, prefix="/api")
