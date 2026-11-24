"""
chat.py — Skema data (Pydantic models) untuk request dan response endpoint /chat.

Digunakan oleh FastAPI untuk validasi input, serialisasi output,
dan pembuatan dokumentasi OpenAPI secara otomatis.
"""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    """Metadata satu sumber dokumen yang diambil dari ChromaDB."""
    source: str           # Nama file sumber (contoh: "Pertemuan 1 - Jaringan Komputer.pdf")
    page: int             # Nomor halaman dalam dokumen tersebut
    score: Optional[float] = None  # Skor relevansi dari similarity search (lebih tinggi = lebih relevan)


class RetrievalInfo(BaseModel):
    """Informasi hasil retrieval — berapa dokumen yang diambil dan dari mana."""
    k: int                        # Jumlah dokumen yang berhasil diambil
    sources: List[SourceMetadata] # Daftar metadata tiap sumber


class ChatRequest(BaseModel):
    """Body request POST /api/chat yang dikirim dari frontend."""
    question: str = Field(..., min_length=1, max_length=1000, description="Pertanyaan mahasiswa")
    mode: Literal["cepat", "berpikir"] = Field("berpikir", description="Mode jawaban: cepat atau berpikir")


class ChatResponse(BaseModel):
    """Response yang dikembalikan oleh endpoint /api/chat."""
    answer: str                          # Teks jawaban dari LLM
    retrieval: RetrievalInfo             # Info dokumen yang digunakan sebagai konteks
    latency_ms: Optional[float] = None   # Waktu pemrosesan total dalam milidetik
