"""
chroma_service.py — Layanan Vector Store ChromaDB untuk Retrieval Dokumen.

Mengelola koneksi ke database vektor ChromaDB dan pencarian kemiripan
semantik (similarity search) berdasarkan embedding model HuggingFace.
"""

import os
import warnings

# Nonaktifkan telemetri analitik internal ChromaDB agar terminal bersih dari pesan error telemetri
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.core.config import settings


class ChromaService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        # Hitung path absolut ke direktori penyimpanan ChromaDB
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        persist_dir = os.path.join(base_dir, "data", "chroma")

        self.vectorstore = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )

    def search(self, query: str, k: int = settings.retrieval_k):
        """Lakukan pencarian semantik untuk mengambil Top-K chunk dokumen beserta skor relevansi."""
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return docs_and_scores


# Singleton instance — diimpor di seluruh aplikasi backend
chroma_service = ChromaService()
