"""
llm_service.py — Layanan generasi jawaban menggunakan LLM (Large Language Model).

Mendukung dua mode:
  - "cepat"    : menggunakan Groq (model ringan, respons cepat)
  - "berpikir" : menggunakan Gemini (model lebih dalam, jawaban lengkap)

Alur kerja: prompt template → LLM → parser → teks jawaban (LangChain pipeline).
"""

import os
from typing import Literal, Tuple, Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.key_rotator import gemini_key_rotator, mask_key


# Tipe mode yang diizinkan
ModeType = Literal["cepat", "berpikir"]


class LLMService:
    def __init__(self):
        # Instruksi sistem untuk chatbot (persona, aturan, dan batasan domain)
        self.system_instruction = """Anda adalah chatbot pembelajaran untuk mata kuliah Jaringan Komputer.

Tugas Anda adalah membantu mahasiswa memahami materi berdasarkan
konteks dokumen yang diberikan oleh sistem.

ATURAN UTAMA:
1. Gunakan konteks retrieval sebagai sumber utama jawaban.
2. Jangan mengarang fakta yang tidak didukung konteks.
3. Jika konteks tidak cukup untuk menjawab, katakan bahwa informasi
   yang diperlukan tidak tersedia pada materi yang diberikan.
4. Pertanyaan di luar domain pembelajaran Jaringan Komputer harus ditolak
   dengan sopan dan diarahkan kembali ke ruang lingkup sistem.
5. Jelaskan konsep dengan bahasa Indonesia yang mudah dipahami mahasiswa.
6. Untuk proses teknis seperti subnetting, tampilkan langkah yang relevan
   dan jangan menghilangkan asumsi penting."""

        # Bangun chain default saat service pertama kali diinisialisasi
        self.default_chain = self._build_chain_for_mode(settings.default_llm_mode)

    def _build_prompt_for_mode(self, mode: ModeType):
        """Buat ChatPromptTemplate sesuai mode — instruksi berbeda untuk cepat vs berpikir."""
        if mode == "cepat":
            system_instruction = self.system_instruction + "\n\nMODE JAWABAN CEPAT:\n- Jawab singkat, padat, dan langsung ke inti.\n- Prioritaskan jawaban 1-3 paragraf atau beberapa poin singkat.\n- Hindari penjelasan yang tidak relevan atau terlalu panjang.\n- Jika perlu, gunakan poin-poin sederhana, bukan elaborasi berlebihan.\n- Jangan buat tabel kecuali pengguna benar-benar meminta."
            user_instruction = (
                "[CONTEXT]\n{retrieved_context}\n\n[USER QUESTION]\n{query}\n\n[INSTRUCTION]\n"
                "Jawab dengan ringkas, jelas, dan sesuai konteks yang tersedia.\n"
                "Jika informasi tidak ditemukan atau tidak cukup, nyatakan keterbatasannya secara singkat."
            )
        else:
            system_instruction = self.system_instruction + "\n\nMODE JAWABAN DETAIL:\n- Berikan jawaban yang lengkap, jelas, dan mudah dipahami.\n- Jelaskan konsep dengan detail yang cukup untuk mahasiswa.\n- Gunakan struktur logis seperti penjelasan singkat, langkah, contoh, dan kesimpulan.\n- Jika relevan, tampilkan tabel yang rapi untuk membandingkan atau merangkum konsep.\n- Sertakan asumsi penting untuk soal teknis seperti subnetting atau routing."
            user_instruction = (
                "[CONTEXT]\n{retrieved_context}\n\n[USER QUESTION]\n{query}\n\n[INSTRUCTION]\n"
                "Jawab dengan detail dan berdasarkan konteks yang tersedia.\n"
                "Gunakan bahasa Indonesia yang jelas.\n"
                "Jika informasi tidak ditemukan atau tidak cukup, nyatakan keterbatasannya dengan jujur.\n"
                "Jika ada perbandingan atau konsep yang perlu dirangkum, gunakan tabel bila relevan."
            )

        return ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", user_instruction),
        ])

    def _get_api_key_for_mode(self, mode: ModeType) -> str:
        """Ambil API key yang sesuai berdasarkan provider yang digunakan pada mode tersebut."""
        if mode == "cepat":
            return settings.groq_api_key or os.getenv("GROQ_API_KEY") or ""
        # Prioritas Gemini: gunakan key aktif dari rotator (apikey.md), fallback ke .env
        active_label, active_key = gemini_key_rotator.get_current_key()
        if active_key:
            return active_key
        return settings.google_api_key or settings.gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

    def _get_model_for_mode(self, mode: ModeType) -> str:
        """Ambil nama model LLM sesuai mode."""
        if mode == "cepat":
            return settings.fast_llm_model or "openai/gpt-oss-120b"
        return settings.thinking_llm_model or settings.llm_model or "gemini-3.5-flash"
  
    def _get_provider_for_mode(self, mode: ModeType) -> str:
        """Tentukan provider LLM (groq atau gemini) berdasarkan mode."""
        if mode == "cepat":
            return settings.fast_llm_provider or "groq"
        return settings.thinking_llm_provider or settings.llm_provider or "gemini"

    def _build_chain_for_mode(self, mode: ModeType, api_key: str = None):
        """
        Bangun LangChain pipeline: prompt → LLM → string parser.
        Provider "groq" menggunakan ChatOpenAI dengan base_url Groq.
        Provider "gemini" menggunakan ChatGoogleGenerativeAI.
        """
        provider = self._get_provider_for_mode(mode)
        key_to_use = api_key or self._get_api_key_for_mode(mode)
        model_name = self._get_model_for_mode(mode)

        if provider == "groq":
            if not key_to_use:
                raise ValueError("API key Groq belum diatur. Isi GROQ_API_KEY di file .env.")
            llm = ChatOpenAI(
                model=model_name,
                api_key=key_to_use,
                base_url="https://api.groq.com/openai/v1",
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
            )
        else:
            if not key_to_use:
                raise ValueError(
                    "API key Gemini belum diatur. Cantumkan di apikey.md atau isi GOOGLE_API_KEY di .env."
                )
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key_to_use,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_retries=1,
                timeout=45,
            )

        prompt_template = self._build_prompt_for_mode(mode)
        return prompt_template | llm | StrOutputParser()

    def _get_chain_for_mode(self, mode: ModeType, api_key: str = None):
        """Buat chain baru setiap request agar selalu menggunakan konfigurasi terkini."""
        return self._build_chain_for_mode(mode, api_key=api_key)

    def generate_answer(self, query: str, context: str, mode: ModeType = None) -> str:
        """
        Generate answer using LLM based on retrieved context.
        Pada mode 'berpikir', otomatis merotasi API key Gemini dari apikey.md
        jika terjadi limit (429, ResourceExhausted, 503 high demand).
        """
        ans, _ = self.generate_answer_with_meta(query, context, mode=mode)
        return ans

    def generate_answer_with_meta(
        self, query: str, context: str, mode: ModeType = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Sama dengan generate_answer, namun menyertakan metadata eksekusi
        (mode, model, label API key yang digunakan, dan status rotasi).
        """
        current_mode = mode or settings.default_llm_mode

        if current_mode == "berpikir":
            total_keys = len(gemini_key_rotator.keys)
            max_attempts = max(1, total_keys)
            last_exception = None

            for attempt in range(max_attempts):
                label, current_key = gemini_key_rotator.get_current_key()
                try:
                    chain = self._build_chain_for_mode("berpikir", api_key=current_key)
                    resp = chain.invoke({
                        "query": query,
                        "retrieved_context": context,
                    })
                    return resp, {
                        "mode": "berpikir",
                        "provider": "gemini",
                        "model": self._get_model_for_mode("berpikir"),
                        "api_key_label": label,
                        "api_key_masked": mask_key(current_key),
                        "attempt": attempt + 1
                    }
                except Exception as e:
                    last_exception = e
                    if gemini_key_rotator.is_limit_error(e) and attempt < max_attempts - 1:
                        next_label, next_key = gemini_key_rotator.rotate_key(reason=str(e))
                        print(f"\n    [!] Gemini Limit/Quota pada [{label}] ({mask_key(current_key)}).")
                        print(f"    [*] Otomatis beralih ke [{next_label}] ({mask_key(next_key)})...")
                        continue
                    # Jika bukan limit error atau semua key sudah dicoba
                    raise e

            if last_exception:
                raise last_exception
        else:
            chain = self._get_chain_for_mode("cepat")
            resp = chain.invoke({
                "query": query,
                "retrieved_context": context,
            })
            return resp, {
                "mode": "cepat",
                "provider": "groq",
                "model": self._get_model_for_mode("cepat"),
                "api_key_label": "GROQ_API_KEY",
                "attempt": 1
            }


# Singleton instance — diimpor di seluruh aplikasi
llm_service = LLMService()
