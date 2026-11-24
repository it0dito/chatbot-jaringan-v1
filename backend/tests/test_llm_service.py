"""
test_llm_service.py — Unit test untuk LLMService.

Pengujian dilakukan tanpa memanggil API eksternal (Gemini/Groq)
dengan teknik mocking menggunakan FakeChain — chain palsu yang
menangkap input tanpa melakukan network call.
"""

from types import SimpleNamespace

from app.generation.llm_service import LLMService


def test_generate_answer_passes_context_to_prompt_chain():
    """
    Pastikan generate_answer() meneruskan query dan konteks
    dengan benar ke chain yang digunakan.
    """
    # Buat instance LLMService tanpa memanggil __init__ (hindari inisialisasi chain sungguhan)
    service = LLMService.__new__(LLMService)

    captured = {}

    class FakeChain:
        """Chain palsu: hanya menyimpan input yang diterima dan mengembalikan string tetap."""
        def invoke(self, values):
            captured.update(values)
            return "jawaban contoh"

    # Override _get_chain_for_mode agar menggunakan FakeChain
    service._get_chain_for_mode = lambda mode: FakeChain()

    result = service.generate_answer("Apa itu subnet?", "Konteks subnet")

    assert result == "jawaban contoh"
    assert captured == {"query": "Apa itu subnet?", "retrieved_context": "Konteks subnet"}


def test_fast_and_thinking_modes_use_different_prompt_instructions():
    """
    Pastikan prompt mode 'cepat' mengandung kata kunci jawaban singkat,
    dan mode 'berpikir' mengandung kata kunci jawaban detail/tabel.
    """
    service = LLMService()

    fast_prompt = service._build_prompt_for_mode("cepat")
    thinking_prompt = service._build_prompt_for_mode("berpikir")

    # Mode cepat harus mengandung instruksi singkat/ringkas
    assert "singkat" in fast_prompt[0][1].lower()
    assert "ringkas" in fast_prompt[0][1].lower()

    # Mode berpikir harus mengandung instruksi detail dan tabel
    assert "tabel" in thinking_prompt[0][1].lower()
    assert "detail" in thinking_prompt[0][1].lower()
