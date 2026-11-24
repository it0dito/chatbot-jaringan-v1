"""
key_rotator.py — Manajemen dan Rotasi Otomatis Google Gemini API Key.

Membaca daftar API key dari berkas apikey.md di root proyek atau dari .env.
Ketika terjadi limit (HTTP 429, ResourceExhausted, Quota Exceeded, 503 Spike),
sistem secara otomatis merotasi kunci ke API key berikutnya tanpa menghentikan evaluasi.
"""

import os
import re
from typing import List, Dict, Tuple, Optional


def mask_key(key: str) -> str:
    """Masking API key untuk keamanan log (menampilkan awal dan akhir)."""
    if not key or len(key) <= 10:
        return "***"
    return f"{key[:8]}...{key[-5:]}"


class GeminiKeyRotator:
    def __init__(self, apikey_file_path: Optional[str] = None):
        self.keys: List[Dict[str, str]] = []
        self.current_index: int = 0
        self.exhausted_keys: set = set()
        self.rotation_history: List[Dict[str, str]] = []

        # Tentukan lokasi apikey.md
        if apikey_file_path and os.path.exists(apikey_file_path):
            self.file_path = apikey_file_path
        else:
            # Coba cari di direktori proyek
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            possible_paths = [
                os.path.join(base_dir, "apikey.md"),
                os.path.join(os.path.dirname(base_dir), "apikey.md"),
                os.path.join(base_dir, "backend", "apikey.md"),
                r"E:\BISMILLAH SKRIPSI\RAG LLM CHATBOT\SISTEM\chatbot-jaringan-komputer\apikey.md"
            ]
            self.file_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    self.file_path = p
                    break

        self.load_keys()

    def load_keys(self) -> None:
        """Membaca dan memvalidasi semua API key dari apikey.md dan .env."""
        loaded_keys = []
        seen_keys = set()
        seen_labels = {}

        if self.file_path and os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        raw_label = parts[0].strip()
                        key_val = parts[1].strip()
                    else:
                        raw_label = f"key_{len(loaded_keys) + 1}"
                        key_val = parts[0].strip()

                    # Pastikan key tidak kosong dan belum duplikat
                    if key_val and key_val not in seen_keys:
                        # Atasi duplikasi label seperti 'apikey6' yang muncul 2 kali
                        count = seen_labels.get(raw_label, 0) + 1
                        seen_labels[raw_label] = count
                        unique_label = f"{raw_label}_{count}" if count > 1 else raw_label

                        loaded_keys.append({
                            "label": unique_label,
                            "key": key_val,
                            "masked": mask_key(key_val),
                            "source": "apikey.md"
                        })
                        seen_keys.add(key_val)

        # Tambahkan juga key dari environment jika ada dan belum ada di list
        env_google_key = os.getenv("GOOGLE_API_KEY", "").strip()
        env_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

        for env_label, env_val in [("GOOGLE_API_KEY (.env)", env_google_key), ("GEMINI_API_KEY (.env)", env_gemini_key)]:
            if env_val and env_val not in seen_keys:
                loaded_keys.append({
                    "label": env_label,
                    "key": env_val,
                    "masked": mask_key(env_val),
                    "source": ".env"
                })
                seen_keys.add(env_val)

        self.keys = loaded_keys
        self.current_index = 0

    def get_current_key_info(self) -> Dict[str, str]:
        """Mengambil data key yang sedang aktif saat ini."""
        if not self.keys:
            return {"label": "None", "key": "", "masked": "None", "source": "None"}
        return self.keys[self.current_index]

    def get_current_key(self) -> Tuple[str, str]:
        """Mengembalikan tuple (label, key) yang sedang aktif."""
        info = self.get_current_key_info()
        return info["label"], info["key"]

    def rotate_key(self, reason: str = "") -> Tuple[str, str]:
        """
        Merotasi API key ke kunci berikutnya di dalam daftar.
        Mengembalikan (label_baru, key_baru).
        """
        if not self.keys:
            return "None", ""

        old_info = self.keys[self.current_index]
        self.exhausted_keys.add(old_info["key"])

        # Pindah ke indeks berikutnya secara memutar
        next_index = (self.current_index + 1) % len(self.keys)
        self.current_index = next_index
        new_info = self.keys[self.current_index]

        self.rotation_history.append({
            "from_label": old_info["label"],
            "from_key": old_info["masked"],
            "to_label": new_info["label"],
            "to_key": new_info["masked"],
            "reason": reason[:100]
        })

        return new_info["label"], new_info["key"]

    def is_limit_error(self, error: Exception) -> bool:
        """
        Mendeteksi apakah exception merupakan penanda bahwa API key saat ini bermasalah
        (limit 429, quota exceeded, 403 forbidden, 503 high demand, atau timeout)
        sehingga sistem harus otomatis merotasi ke kunci berikutnya dari apikey.md.
        """
        err_str = str(error).lower()
        err_type = type(error).__name__.lower()
        signals = [
            "429",
            "resourceexhausted",
            "resource_exhausted",
            "quota",
            "rate limit",
            "ratelimit",
            "too many requests",
            "exhausted",
            "service unavailable",
            "503",
            "504",
            "high demand",
            "403",
            "forbidden",
            "timed out",
            "timeout",
            "deadline exceeded"
        ]
        return any(sig in err_str for sig in signals) or any(sig in err_type for sig in ["timeout", "resourceexhausted"])

    def print_status(self) -> None:
        """Menampilkan ringkasan seluruh key yang dimuat."""
        print(f"\n[Gemini Key Manager] Terdaftar {len(self.keys)} API Key Gemini:")
        for idx, item in enumerate(self.keys):
            active_marker = "==> [AKTIF]" if idx == self.current_index else "   "
            exhausted_marker = "(Limit)" if item["key"] in self.exhausted_keys else ""
            print(f"  {active_marker} #{idx+1}. {item['label']} : {item['masked']} {exhausted_marker}")
        print()


# Singleton instance
gemini_key_rotator = GeminiKeyRotator()
