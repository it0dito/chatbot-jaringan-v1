"""
ingest.py — Script pipeline ingestion dokumen ke ChromaDB.

Alur kerja:
  1. Baca semua file PDF dari folder data/documents/
  2. Ekstrak teks per halaman menggunakan pypdf
  3. Pecah teks menjadi chunk-chunk kecil (RecursiveCharacterTextSplitter)
  4. Buat embedding vektor menggunakan model HuggingFace
  5. Simpan embedding ke ChromaDB untuk digunakan saat retrieval

Jalankan sekali sebelum menjalankan API:
  python backend/scripts/ingest.py
"""

import os
import sys
from pypdf import PdfReader
from typing import List
from dotenv import load_dotenv

# Tambahkan direktori parent ke sys.path agar bisa import dari package `app`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Muat variabel dari .env (diperlukan untuk konfigurasi settings)
load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """
    Ekstrak teks dari setiap halaman PDF dan kembalikan sebagai list LangChain Document.
    Setiap Document menyimpan teks halaman beserta metadata (nama file, nomor halaman, dll.).
    """
    documents = []
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            # Lewati halaman yang kosong atau hanya berisi whitespace
            if text and text.strip():
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "page": page_num + 1,
                    "domain": "jaringan_komputer",
                    "document_type": "modul",
                    "ingestion_version": "v1"
                }
                documents.append(Document(page_content=text, metadata=metadata))
        print(f"Extracted {len(documents)} pages from {os.path.basename(pdf_path)}")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return documents


def ingest_documents():
    """
    Pipeline utama ingestion:
    1. Kumpulkan semua PDF → 2. Chunking → 3. Embedding → 4. Simpan ke ChromaDB
    """
    documents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")

    if not os.path.exists(documents_dir):
        print(f"Directory {documents_dir} does not exist.")
        return

    # Langkah 1: Baca semua PDF dari folder documents
    all_docs = []
    for filename in os.listdir(documents_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(documents_dir, filename)
            all_docs.extend(extract_text_from_pdf(file_path))

    if not all_docs:
        print("No documents found to ingest.")
        return

    # Langkah 2: Chunking — pecah dokumen menjadi potongan kecil yang saling tumpang tindih
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,       # Panjang maksimal satu chunk (karakter)
        chunk_overlap=settings.chunk_overlap, # Overlap agar konteks tidak terputus di batas chunk
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(all_docs)

    # Tambahkan indeks chunk ke metadata untuk keperluan tracing
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"Created {len(chunks)} chunks.")

    # Langkah 3: Load embedding model dari HuggingFace
    print(f"Loading embedding model {settings.embedding_model}...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    # Langkah 4: Simpan chunk + embedding ke ChromaDB
    persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
    print(f"Saving to ChromaDB at {persist_directory}...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.chroma_collection,
        persist_directory=persist_directory
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_documents()
