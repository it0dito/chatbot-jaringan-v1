"""
endpoints.py — Definisi semua route/endpoint HTTP API chatbot.

Berisi tiga endpoint utama:
  - GET  /api/        → cek status API
  - POST /api/chat    → menerima pertanyaan, menjalankan pipeline RAG, mengembalikan jawaban
  - GET  /api/health  → health check untuk monitoring
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas.chat import ChatRequest, ChatResponse, RetrievalInfo, SourceMetadata
from app.retrieval.chroma_service import chroma_service
from app.generation.llm_service import llm_service
import time

router = APIRouter()


# --- Endpoint root API ---
@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def api_root():
    return {"status": "ok", "message": "Chatbot API is running", "endpoints": ["/api/chat", "/api/health"]}


# --- Endpoint utama: menerima pertanyaan dan mengembalikan jawaban RAG ---
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()

    # Validasi: pastikan pertanyaan tidak kosong
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")

    try:
        # 1. Retrieval — cari dokumen relevan dari ChromaDB berdasarkan query
        retrieved_docs = chroma_service.search(query)

        context_texts = []
        sources = []

        # 2. Kumpulkan teks konteks dan metadata sumber dari setiap dokumen yang ditemukan
        for doc, score in retrieved_docs:
            context_texts.append(doc.page_content)
            src_meta = SourceMetadata(
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page", 0),
                score=float(score)
            )
            sources.append(src_meta)

        # 3. Gabungkan semua teks konteks menjadi satu string
        combined_context = "\n\n---\n\n".join(context_texts)

        # 4. Generation — kirim query + konteks ke LLM untuk menghasilkan jawaban
        answer = llm_service.generate_answer(query, combined_context, mode=request.mode)

        # Hitung latency total (ms)
        latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            answer=answer,
            retrieval=RetrievalInfo(
                k=len(sources),
                sources=sources
            ),
            latency_ms=latency_ms
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- Health check: digunakan untuk memverifikasi API berjalan normal ---
@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "Chatbot RAG API is healthy"}


@router.get("/status", include_in_schema=False)
async def status_check():
    return {"status": "ok", "message": "Chatbot RAG API is healthy"}
