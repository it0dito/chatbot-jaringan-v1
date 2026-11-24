/**
 * api.ts — Layanan komunikasi frontend ke backend API.
 *
 * Menyediakan:
 *  - Interface TypeScript untuk struktur data request/response
 *  - Fungsi `sendChatMessage()` untuk mengirim pertanyaan ke endpoint /api/chat
 *  - Penanganan error HTTP dengan pesan yang informatif
 */

/** Metadata satu sumber dokumen yang dikembalikan oleh backend. */
export interface SourceMetadata {
  source: string;  // Nama file PDF sumber
  page: number;    // Nomor halaman dalam file tersebut
  score: number;   // Skor relevansi (0–1, lebih tinggi = lebih relevan)
}

/** Struktur lengkap response dari endpoint POST /api/chat. */
export interface ChatResponse {
  answer: string;           // Teks jawaban dari LLM
  retrieval: {
    k: number;              // Jumlah dokumen yang diambil saat retrieval
    sources: SourceMetadata[];  // Daftar sumber yang digunakan sebagai konteks
  };
  latency_ms: number;       // Waktu pemrosesan total (milidetik)
}

// URL API — baca dari variabel environment Vite, fallback ke localhost
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

/** Mode jawaban yang tersedia: ringkas (Groq) atau mendalam (Gemini). */
export type ChatMode = "cepat" | "berpikir";

/**
 * Kirim pertanyaan ke backend dan kembalikan response chatbot.
 *
 * @param message  - Teks pertanyaan dari pengguna
 * @param mode     - Mode jawaban ("cepat" atau "berpikir"), default "berpikir"
 * @returns        ChatResponse berisi jawaban, sumber dokumen, dan latency
 * @throws         Error dengan pesan dari server jika request gagal
 */
export async function sendChatMessage(message: string, mode: ChatMode = "berpikir"): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: message, mode }),
    });

    // Jika server mengembalikan error, ekstrak pesan dari body response
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));


      let errorMsg = `Server error: ${response.status}`;
      if (errorData && errorData.detail) {
        if (typeof errorData.detail === 'string') {
          errorMsg = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          // FastAPI validation errors dikembalikan sebagai array
          errorMsg = errorData.detail.map((e: any) => e.msg).join(', ');
        } else {
          errorMsg = JSON.stringify(errorData.detail);
        }
      }


      throw new Error(errorMsg);
    }

    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}
