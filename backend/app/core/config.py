"""
config.py — Konfigurasi global aplikasi menggunakan Pydantic Settings.

Semua nilai konfigurasi dibaca dari file .env secara otomatis.
Gunakan objek `settings` (singleton) di seluruh aplikasi untuk
mengakses konfigurasi tanpa perlu membaca .env berulang kali.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Informasi Aplikasi ---
    app_name: str = "chatbot-jaringan-komputer"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # URL frontend untuk konfigurasi CORS
    frontend_url: str = "http://localhost:5173"

    # --- Konfigurasi Embedding ---
    embedding_model: str = "intfloat/multilingual-e5-small"  # Model HuggingFace untuk embedding teks
    chunk_size: int = 1000       # Ukuran maksimal satu chunk dokumen (karakter)
    chunk_overlap: int = 200     # Overlap antar chunk agar konteks tidak terputus

    # --- Konfigurasi Vector Store (ChromaDB) ---
    vector_db: str = "chroma"
    chroma_persist_directory: str = "./data/chroma"      # Lokasi penyimpanan ChromaDB
    chroma_collection: str = "jaringan-komputer"         # Nama koleksi di ChromaDB
    retrieval_k: int = 5                                  # Jumlah dokumen yang diambil saat retrieval

    # --- Konfigurasi LLM ---
    default_llm_mode: str = "berpikir"        # Mode default: "cepat" (Groq) atau "berpikir" (Gemini)
    llm_provider: str = "gemini"
    llm_model: str = "gemini-flash-latest"
    fast_llm_provider: str = "groq"           # Provider untuk mode cepat
    fast_llm_model: str = "openai/gpt-oss-120b"
    thinking_llm_provider: str = "gemini"     # Provider untuk mode berpikir
    thinking_llm_model: str = "gemini-flash-latest"
    llm_temperature: float = 0.1              # Tingkat kreativitas jawaban (0 = deterministik)
    llm_top_p: float = 0.9

    # --- API Keys (diisi di file .env, jangan di-hardcode) ---
    google_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # Konfigurasi Pydantic: baca dari file .env, abaikan field yang tidak dikenal
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton — digunakan di seluruh aplikasi: from app.core.config import settings
settings = Settings()
