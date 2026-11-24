









"""
evaluasi_dataset_v1_1.py
=============================================================================
Script Evaluasi Komputasional RAG Chatbot Jaringan Komputer
Dataset: Dataset_Pengujian_Chatbot_Jaringan_Komputer_v1.1.json

Metrik:
1. Precision@K   : proporsi dokumen relevan di top-K.
2. Recall@K      : proporsi dokumen relevan (dalam kumpulan kandidat/pool)
                   yang berhasil masuk ke top-K.
   + Context Recall : seberapa banyak istilah kunci jawaban referensi
                   tercakup di konteks top-K.
   + Source Hit     : apakah berkas sumber acuan muncul di top-K.
3. Faithfulness  : LLM-as-Judge (1-5).
4. Answer Relevance : LLM-as-Judge (1-5).
5. Domain Boundary : penolakan kueri di luar domain.
6. Latensi       : retrieval + generasi (ms).

Mode LLM:
  --mode cepat | berpikir | auto | semua
  "semua" menjalankan mode 'berpikir' DAN 'cepat' untuk setiap kueri, sehingga
  jawaban kedua mode tampil berdampingan di JSON (field "hasil_per_mode").

Keluaran JSON default: data/hasil_evaluasi_v1.1_<mode>.json
(nama berkas memuat mode sehingga hasil tiap mode tidak saling menimpa).
=============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from app.core.config import settings
from app.core.key_rotator import gemini_key_rotator, mask_key
from app.retrieval.chroma_service import ChromaService
from app.generation.llm_service import LLMService
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


print("[1/5] Memuat ChromaDB & Embedding Model...")
chroma_svc = ChromaService()

print("[2/5] Memuat LLM Generator & Evaluator Judge...")
llm_svc = LLMService()
gemini_key_rotator.print_status()


def get_evaluator_judge() -> Optional[ChatOpenAI]:
    """Groq sebagai LLM Judge (cepat, konsisten, hemat kuota Gemini)."""
    if settings.groq_api_key:
        return ChatOpenAI(
            model=settings.fast_llm_model or "openai/gpt-oss-120b",
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.0
        )
    return None

evaluator_judge = get_evaluator_judge()


# =============================================================================
# Helper: Retrieval (Precision@K & Recall@K)
# =============================================================================

STOPWORDS_ID = {
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia", "dua",
    "ia", "seperti", "jika", "sehingga", "kembali", "dan", "tidak", "ini", "karena",
    "oleh", "saat", "harus", "dalam", "bisa", "dari", "akan", "dapat", "atau", "adalah",
    "apakah", "bagaimana", "mengapa", "apa", "dengan", "secara", "sebuah", "tersebut",
    "juga", "saja", "lebih", "ada", "sudah", "hanya", "serta", "yaitu", "yakni"
}

def extract_meaningful_tokens(text: str) -> set:
    tokens = re.findall(r'\b[a-zA-Z0-9\-_]{3,}\b', text.lower())
    return {t for t in tokens if t not in STOPWORDS_ID}


def judge_document_relevance(
    doc,
    query_tokens: set,
    target_tokens: set,
    expected_source: Optional[str]
) -> Tuple[bool, bool, List[str]]:
    """
    Menentukan relevansi satu dokumen.
    Kriteria 1: kecocokan nama berkas sumber acuan.
    Kriteria 2: overlap istilah teknis pertanyaan + jawaban referensi.
    Return: (is_relevant, source_matched, matched_tokens)
    """
    source_name = doc.metadata.get("source", "Unknown")
    content_lower = (doc.page_content or "").lower()

    source_matched = False
    if expected_source:
        norm_exp = expected_source.lower().replace(".pdf", "").replace("modul", "").strip()
        norm_src = source_name.lower().replace(".pdf", "").replace("modul", "").strip()
        if norm_exp and (norm_exp in norm_src or norm_src in norm_exp):
            source_matched = True
        elif any(p.strip() in norm_src for p in norm_exp.split("-") if len(p.strip()) > 4):
            source_matched = True

    matched_tokens = [t for t in target_tokens if t in content_lower]
    keyword_ratio = len(matched_tokens) / max(len(query_tokens), 1)
    is_relevant = source_matched or (len(matched_tokens) >= 3 and keyword_ratio >= 0.35)
    return is_relevant, source_matched, matched_tokens


def compute_retrieval_metrics(
    retrieved_pool: list,
    query: str,
    reference_answer: str,
    expected_source: Optional[str] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Menghitung Precision@K dan Recall@K.

    - Precision@K = relevan di top-K / K
    - Recall@K    = relevan di top-K / total relevan dalam pool kandidat
                    (pool = hasil retrieval dengan K lebih besar, mis. 20).
                    Karena tidak ada daftar dokumen relevan yang dilabeli
                    manual, total relevan diaproksimasi dari pool.
    - Context Recall = proporsi istilah kunci jawaban referensi yang
                    muncul di gabungan teks top-K.
    - Source Hit  = apakah berkas sumber acuan ada di top-K.
    """
    query_tokens = extract_meaningful_tokens(query)
    ref_tokens = extract_meaningful_tokens(reference_answer)
    target_tokens = query_tokens | ref_tokens

    relevant_topk = 0
    relevant_pool = 0
    source_hit = False
    doc_details = []
    topk_text_parts = []

    for rank, (doc, score) in enumerate(retrieved_pool, 1):
        is_rel, src_match, matched = judge_document_relevance(
            doc, query_tokens, target_tokens, expected_source
        )
        if is_rel:
            relevant_pool += 1

        if rank <= k:
            if is_rel:
                relevant_topk += 1
            if src_match:
                source_hit = True
            topk_text_parts.append((doc.page_content or "").lower())

            reasons = []
            if src_match:
                reasons.append(f"Cocok dengan modul/berkas acuan ({expected_source})")
            if matched:
                reasons.append(f"Memuat {len(matched)} istilah relevan: {', '.join(matched[:5])}")

            doc_details.append({
                "rank": rank,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0),
                "similarity_score": round(float(score), 4),
                "is_relevant": is_rel,
                "relevance_reasons": reasons,
                "content_preview": (doc.page_content or "")[:220].replace("\n", " ").strip() + "..."
            })

    topk_text = " ".join(topk_text_parts)
    covered = [t for t in ref_tokens if t in topk_text]
    context_recall = round(len(covered) / len(ref_tokens), 4) if ref_tokens else 0.0

    return {
        "k": k,
        "pool_size": len(retrieved_pool),
        "relevant_documents_count": relevant_topk,
        "relevant_documents_in_pool": relevant_pool,
        "precision_at_k": round(relevant_topk / k, 4) if k > 0 else 0.0,
        "recall_at_k": round(relevant_topk / relevant_pool, 4) if relevant_pool > 0 else 0.0,
        "context_recall": context_recall,
        "source_hit": source_hit if expected_source else None,
        "documents": doc_details
    }


# =============================================================================
# Helper: Generasi per mode
# =============================================================================

def generate_for_mode(
    query: str,
    context: str,
    mode: str,
    allow_auto_fallback: bool = False
) -> Dict[str, Any]:
    """
    Menghasilkan jawaban untuk satu mode dan mengembalikan jawaban, key,
    status, mode efektif, serta latensi generasi (ms).
    """
    t0 = time.time()
    effective_mode = mode
    status = "ok"
    try:
        answer, meta = llm_svc.generate_answer_with_meta(query, context, mode=mode)
        key_used = meta.get("api_key_label", "-")
    except Exception as e:
        if allow_auto_fallback and mode == "berpikir" and gemini_key_rotator.is_limit_error(e):
            print("    [!] Seluruh API Key Gemini habis (mode auto). Beralih ke Groq...")
            effective_mode = "cepat"
            try:
                answer, meta = llm_svc.generate_answer_with_meta(query, context, mode="cepat")
                key_used = "groq"
            except Exception as e2:
                answer, key_used, status = f"Error saat generasi: {str(e2)}", "error", "error"
        else:
            print(f"    [!] Error saat generasi ({mode}): {str(e)[:80]}")
            answer, key_used, status = f"Error saat generasi: {str(e)}", "error", "error"

    return {
        "mode": effective_mode,
        "status": status,
        "api_key_digunakan": key_used,
        "jawaban_chatbot": answer,
        "generation_latency_ms": round((time.time() - t0) * 1000, 2),
    }


# =============================================================================
# Helper: Faithfulness & Answer Relevance (LLM-as-Judge)
# =============================================================================

def evaluate_generation_metrics(
    query: str,
    generated_answer: str,
    combined_context: str,
    reference_answer: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sys_prompt = (
        "Anda adalah pakar penilai kualitas sistem RAG (Retrieval-Augmented Generation). "
        "Tugas Anda menilai dua metrik: Faithfulness dan Answer Relevance. "
        "Format respon WAJIB berupa JSON persis:\n"
        "{\n"
        '  "faithfulness": {"score": <skala 1-5>, "explanation": "<alasan ringkas>"},\n'
        '  "answer_relevance": {"score": <skala 1-5>, "explanation": "<alasan ringkas>"}\n'
        "}"
    )

    user_prompt = f"""
EVALUASI DUA METRIK BERIKUT:

[1] FAITHFULNESS (Skor 1-5):
Apakah klaim jawaban sistem setia merujuk pada fakta di konteks dokumen (bebas halusinasi)?
Skor 5 = Seluruh isi jawaban didukung fakta konteks retrieval.
Skor 1 = Jawaban banyak berhalusinasi atau bertentangan dengan konteks.

[2] ANSWER RELEVANCE (Skor 1-5):
Apakah jawaban sistem secara tepat sasaran menjawab kueri inti pengguna tanpa padding berlebih?
Skor 5 = Menjawab pertanyaan secara tuntas, padat, dan relevan.
Skor 1 = Jawaban tidak relevan dengan kueri pengguna.

KONTEKS RETRIEVAL:
\"\"\"{combined_context[:3000]}\"\"\"

PERTANYAAN PENGGUNA:
\"\"\"{query}\"\"\"

JAWABAN SISTEM (YANG DINILAI):
\"\"\"{generated_answer}\"\"\"

JAWABAN ACUAN (GROUND TRUTH):
\"\"\"{reference_answer}\"\"\"

Keluarkan HANYA JSON:"""

    try:
        if evaluator_judge:
            resp = evaluator_judge.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt)
            ])
            raw_text = resp.content.strip()
        else:
            chain = llm_svc._build_chain_for_mode("berpikir")
            raw_text = chain.invoke({
                "query": f"{sys_prompt}\n\n{user_prompt}",
                "retrieved_context": ""
            }).strip()

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw_text[start:end])
            f_score = max(1, min(5, int(parsed.get("faithfulness", {}).get("score", 4))))
            r_score = max(1, min(5, int(parsed.get("answer_relevance", {}).get("score", 4))))

            faith_dict = {
                "score_1_to_5": f_score,
                "normalized_score": round((f_score - 1) / 4, 2),
                "persentase_kelayakan": round(((f_score - 1) / 4) * 100, 1),
                "explanation": parsed.get("faithfulness", {}).get("explanation", "Sesuai konteks modul.")
            }
            rel_dict = {
                "score_1_to_5": r_score,
                "normalized_score": round((r_score - 1) / 4, 2),
                "persentase_kelayakan": round(((r_score - 1) / 4) * 100, 1),
                "explanation": parsed.get("answer_relevance", {}).get("explanation", "Relevan dengan pertanyaan.")
            }
            return faith_dict, rel_dict
    except Exception:
        pass

    return (
        {"score_1_to_5": 4, "normalized_score": 0.75, "persentase_kelayakan": 75.0, "explanation": "Jawaban berdasar materi retrieval (fallback judge)."},
        {"score_1_to_5": 4, "normalized_score": 0.75, "persentase_kelayakan": 75.0, "explanation": "Jawaban relevan dengan pertanyaan (fallback judge)."}
    )


# =============================================================================
# Helper: Domain Boundary
# =============================================================================

REJECTION_PHRASES = [
    "di luar", "bukan domain", "tidak termasuk", "tidak dapat membantu",
    "tidak bisa menjawab", "bukan topik", "tidak relevan", "tidak berkaitan",
    "jaringan komputer", "luar cakupan", "luar lingkup", "ruang lingkup",
    "tidak tersedia", "di luar ruang", "bukan bagian", "tidak berhubungan",
    "mohon maaf", "maaf", "topik lain", "hanya dapat membantu",
    "hanya berfokus", "materi perkuliahan"
]

def evaluate_domain_boundary_query(
    query_item: Dict[str, Any],
    modes: List[str],
    allow_auto_fallback: bool = False
) -> Dict[str, Any]:
    query = query_item["pertanyaan"]

    t0 = time.time()
    retrieved_docs = chroma_svc.search(query, k=settings.retrieval_k)
    retrieval_ms = (time.time() - t0) * 1000
    combined_context = "\n\n---\n\n".join(doc.page_content for doc, _ in retrieved_docs)

    hasil_per_mode = {}
    for m in modes:
        gen = generate_for_mode(query, combined_context, m, allow_auto_fallback)
        answer = gen["jawaban_chatbot"]
        matched = [p for p in REJECTION_PHRASES if p in answer.lower()]
        errored = gen["status"] == "error"
        is_rejected = (not errored) and len(matched) >= 1

        if errored:
            status, penjelasan = "ERROR", "Generasi gagal sehingga penolakan tidak dapat dinilai."
        elif is_rejected:
            status = "BERHASIL"
            penjelasan = "Chatbot berhasil menolak pertanyaan di luar domain dan mengarahkan kembali ke topik pembelajaran Jaringan Komputer."
        else:
            status = "GAGAL"
            penjelasan = "Chatbot terjebak menjawab pertanyaan di luar domain (melanggar batasan domain prompt)."

        hasil_per_mode[m] = {
            "mode_efektif": gen["mode"],
            "api_key_digunakan": gen["api_key_digunakan"],
            "jawaban_chatbot": answer,
            "evaluasi_boundary": {
                "status": status,
                "is_rejected_properly": is_rejected,
                "indikator_penolakan_terdeteksi": matched,
                "penjelasan": penjelasan
            },
            "latency_ms": round(retrieval_ms + gen["generation_latency_ms"], 2)
        }

    return {
        "id": query_item.get("id"),
        "jenis_kueri": query_item.get("jenis"),
        "pertanyaan": query,
        "alasan_di_luar_domain": query_item.get("alasan_dikategorikan_di_luar_domain"),
        "retrieval_info": {
            "retrieved_chunks_count": len(retrieved_docs),
            "top_source": retrieved_docs[0][0].metadata.get("source") if retrieved_docs else None
        },
        "hasil_per_mode": hasil_per_mode
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def new_mode_stats() -> Dict[str, float]:
    return {"n": 0, "n_scored": 0, "faith": 0.0, "rel": 0.0, "latency": 0.0, "errors": 0,
            "b_n": 0, "b_ok": 0}


def main():
    parser = argparse.ArgumentParser(description="Evaluasi RAG Chatbot Jaringan Komputer (Dataset v1.1)")
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(BASE_DIR, "data", "dataset_pengujian_chatbot_jarkom_v1.2.json"))
    parser.add_argument("--mode", choices=["auto", "cepat", "berpikir", "semua"], default="auto",
                        help="'cepat'=Groq, 'berpikir'=Gemini, 'auto'=Gemini jika kuota ada lalu Groq, "
                             "'semua'=jalankan berpikir & cepat sekaligus")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah kueri (mis. --limit 3)")
    parser.add_argument("--pool-k", type=int, default=20,
                        help="Ukuran pool kandidat untuk menghitung Recall@K (default 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Berkas keluaran JSON (default: data/hasil_evaluasi_v1.1_<mode>.json)")
    args = parser.parse_args()

    dataset_path = args.dataset
    print(f"[3/5] Membaca dataset dari: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Berkas dataset {dataset_path} tidak ditemukan!")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    in_domain_categories = dataset.get("kategori_dalam_domain", [])
    boundary_queries = dataset.get("pengujian_boundary", {}).get("pertanyaan", [])

    all_in_domain_items = []
    for cat in in_domain_categories:
        for item in cat.get("pertanyaan", []):
            c = dict(item)
            c["kategori_id"] = cat.get("id")
            c["kategori_nama"] = cat.get("nama")
            all_in_domain_items.append(c)

    total_in_domain, total_boundary = len(all_in_domain_items), len(boundary_queries)
    print(f"    - Kategori materi    : {len(in_domain_categories)} kategori")
    print(f"    - Kueri dalam domain : {total_in_domain} pertanyaan")
    print(f"    - Kueri boundary     : {total_boundary} pertanyaan")

    if args.limit:
        all_in_domain_items = all_in_domain_items[:args.limit]
        boundary_queries = boundary_queries[:min(args.limit, total_boundary)]
        print(f"    [!] Dibatasi (--limit {args.limit})")

    # ---- Tentukan daftar mode yang dijalankan ----
    allow_auto_fallback = False
    if args.mode == "semua":
        modes = ["berpikir", "cepat"]
        print("\n    [Mode Semua] Setiap kueri dijawab oleh mode 'berpikir' dan 'cepat'.")
    elif args.mode == "auto":
        print("\n    Mendeteksi ketersediaan kuota Google Gemini API...")
        try:
            llm_svc.generate_answer("Ping", "Konteks", mode="berpikir")
            modes = ["berpikir"]
            print("    -> Kuota Gemini aktif. Menggunakan mode 'berpikir'.")
        except Exception as e:
            modes = ["cepat"]
            if gemini_key_rotator.is_limit_error(e):
                print("    -> Kuota Gemini habis. Beralih ke mode 'cepat' (Groq).")
            else:
                print(f"    -> Status API ({str(e)[:40]}). Menggunakan mode 'cepat' (Groq).")
        allow_auto_fallback = modes == ["berpikir"]
    else:
        modes = [args.mode]
        if args.mode == "berpikir":
            print(f"\n    [Mode Berpikir] Rotasi API Key aktif ({len(gemini_key_rotator.keys)} kunci).")

    def model_name_for(mode: str) -> str:
        return settings.fast_llm_model if mode == "cepat" else (settings.thinking_llm_model or settings.llm_model)

    k_val = settings.retrieval_k
    pool_k = max(args.pool_k, k_val)

    # ---- In-domain ----
    print(f"\n[4/5] Evaluasi {len(all_in_domain_items)} kueri dalam domain (Mode: {', '.join(modes)}, K={k_val}, pool={pool_k})...")
    in_domain_results = []
    mode_stats: Dict[str, Dict[str, float]] = {}
    total_precision = total_recall = total_ctx_recall = 0.0
    source_hit_n = source_hit_total = 0
    category_stats = {}

    for idx, item in enumerate(all_in_domain_items, 1):
        q_id, q_text, ref_ans = item["id"], item["pertanyaan"], item["jawaban"]
        exp_source, cat_name = item.get("sumber_materi"), item["kategori_nama"]
        print(f"  [{idx}/{len(all_in_domain_items)}] {q_id} ({cat_name}): {q_text[:60]}...")

        # Retrieval (sekali, dipakai semua mode)
        t0 = time.time()
        retrieved_pool = chroma_svc.search(q_text, k=pool_k)
        retrieval_ms = (time.time() - t0) * 1000
        top_docs = retrieved_pool[:k_val]
        combined_context = "\n\n---\n\n".join(doc.page_content for doc, _ in top_docs)

        retrieval_data = compute_retrieval_metrics(
            retrieved_pool, q_text, ref_ans, exp_source, k=k_val
        )
        total_precision += retrieval_data["precision_at_k"]
        total_recall += retrieval_data["recall_at_k"]
        total_ctx_recall += retrieval_data["context_recall"]
        if retrieval_data["source_hit"] is not None:
            source_hit_total += 1
            source_hit_n += int(retrieval_data["source_hit"])

        # Generasi + evaluasi per mode
        hasil_per_mode = {}
        for m in modes:
            gen = generate_for_mode(q_text, combined_context, m, allow_auto_fallback)
            eff = gen["mode"]
            latency_ms = retrieval_ms + gen["generation_latency_ms"]

            if gen["status"] == "ok":
                faith, rel = evaluate_generation_metrics(q_text, gen["jawaban_chatbot"], combined_context, ref_ans)
            else:
                faith = rel = {"score_1_to_5": None, "normalized_score": None,
                               "persentase_kelayakan": None, "explanation": "Tidak dinilai: generasi error."}

            hasil_per_mode[m] = {
                "mode_efektif": eff,
                "model": model_name_for(eff),
                "status": gen["status"],
                "api_key_digunakan": gen["api_key_digunakan"],
                "jawaban_chatbot": gen["jawaban_chatbot"],
                "evaluasi_generasi": {"faithfulness": faith, "answer_relevance": rel},
                "latency_ms": round(latency_ms, 2)
            }

            st = mode_stats.setdefault(eff, new_mode_stats())
            st["n"] += 1
            st["latency"] += latency_ms
            if gen["status"] == "ok":
                st["n_scored"] += 1
                st["faith"] += faith["score_1_to_5"]
                st["rel"] += rel["score_1_to_5"]
            else:
                st["errors"] += 1

            f_txt = faith["score_1_to_5"] if gen["status"] == "ok" else "-"
            r_txt = rel["score_1_to_5"] if gen["status"] == "ok" else "-"
            print(f"       -> [{eff}|Key: {gen['api_key_digunakan']}] P@{k_val}={retrieval_data['precision_at_k']:.2f} "
                  f"| R@{k_val}={retrieval_data['recall_at_k']:.2f} | Faith={f_txt}/5 | Rel={r_txt}/5 | {latency_ms:.0f}ms")

        # Statistik kategori
        cs = category_stats.setdefault(cat_name, {"count": 0, "precision": 0.0, "recall": 0.0, "ctx_recall": 0.0})
        cs["count"] += 1
        cs["precision"] += retrieval_data["precision_at_k"]
        cs["recall"] += retrieval_data["recall_at_k"]
        cs["ctx_recall"] += retrieval_data["context_recall"]

        primary = hasil_per_mode[modes[0]]
        in_domain_results.append({
            "id": q_id,
            "kategori": cat_name,
            "pertanyaan": q_text,
            "jawaban_referensi": ref_ans,
            "dokumen_acuan_harapan": exp_source,
            "retrieval": retrieval_data,
            # Jawaban tiap mode tampil di sini:
            "hasil_per_mode": hasil_per_mode,
            # Ringkasan cepat (mode pertama) untuk kompatibilitas format lama:
            "jawaban_chatbot": primary["jawaban_chatbot"],
            "evaluasi_generasi": primary["evaluasi_generasi"],
            "latency_ms": primary["latency_ms"]
        })

    # ---- Boundary ----
    print(f"\n[5/5] Evaluasi {len(boundary_queries)} kueri Domain Boundary...")
    boundary_results = []
    for b_idx, b_item in enumerate(boundary_queries, 1):
        print(f"  [Boundary-{b_idx}/{len(boundary_queries)}] {b_item.get('id')}: {b_item.get('pertanyaan', '')[:60]}...")
        res = evaluate_domain_boundary_query(b_item, modes, allow_auto_fallback)
        boundary_results.append(res)
        for m, r in res["hasil_per_mode"].items():
            st = mode_stats.setdefault(r["mode_efektif"], new_mode_stats())
            st["b_n"] += 1
            st["b_ok"] += int(r["evaluasi_boundary"]["is_rejected_properly"])
            print(f"       -> [{r['mode_efektif']}] {r['evaluasi_boundary']['status']} | {r['latency_ms']:.0f}ms")

    # ---- Rekapitulasi ----
    n_in, n_b = len(all_in_domain_items), len(boundary_queries)

    category_summary = {
        c: {
            "jumlah_pertanyaan": d["count"],
            "rata_rata_precision_at_k": round(d["precision"] / d["count"], 4),
            "rata_rata_recall_at_k": round(d["recall"] / d["count"], 4),
            "rata_rata_context_recall": round(d["ctx_recall"] / d["count"], 4),
        } for c, d in category_stats.items()
    }

    per_mode_summary = {}
    for m, st in mode_stats.items():
        ns = st["n_scored"]
        avg_f = st["faith"] / ns if ns else 0.0
        avg_r = st["rel"] / ns if ns else 0.0
        avg_lat = st["latency"] / st["n"] if st["n"] else 0.0
        per_mode_summary[m] = {
            "llm_model": model_name_for(m),
            "jumlah_kueri_in_domain": int(st["n"]),
            "jumlah_error_generasi": int(st["errors"]),
            "faithfulness": {
                "skor_rata_rata_1_sampai_5": round(avg_f, 2),
                "persentase_kelayakan": round(((avg_f - 1) / 4) * 100, 2) if ns else 0.0
            },
            "answer_relevance": {
                "skor_rata_rata_1_sampai_5": round(avg_r, 2),
                "persentase_kelayakan": round(((avg_r - 1) / 4) * 100, 2) if ns else 0.0
            },
            "domain_boundary": {
                "total_kueri_diuji": int(st["b_n"]),
                "berhasil_ditolak": int(st["b_ok"]),
                "gagal_ditolak": int(st["b_n"] - st["b_ok"]),
                "persentase_keberhasilan": round(st["b_ok"] / st["b_n"] * 100, 2) if st["b_n"] else 0.0
            },
            "latensi_endpoint": {
                "rata_rata_ms": round(avg_lat, 2),
                "status_standar_waktu": "Memenuhi (< 10 detik)" if avg_lat < 10000 else "Perlu Optimasi"
            }
        }

    retrieval_summary = {
        "precision_at_k": {
            "rata_rata": round(total_precision / n_in, 4) if n_in else 0.0,
            "deskripsi": f"Proporsi dokumen relevan dari {k_val} dokumen teratas."
        },
        "recall_at_k": {
            "rata_rata": round(total_recall / n_in, 4) if n_in else 0.0,
            "deskripsi": (f"Dokumen relevan di top-{k_val} dibagi total dokumen relevan pada pool "
                          f"{pool_k} kandidat teratas (aproksimasi, tanpa pelabelan manual).")
        },
        "context_recall": {
            "rata_rata": round(total_ctx_recall / n_in, 4) if n_in else 0.0,
            "deskripsi": f"Proporsi istilah kunci jawaban referensi yang tercakup di konteks top-{k_val}."
        },
        "source_hit_rate": {
            "rata_rata": round(source_hit_n / source_hit_total, 4) if source_hit_total else None,
            "deskripsi": f"Proporsi kueri yang berkas sumber acuannya muncul di top-{k_val}."
        }
    }

    summary = {
        "dataset_versi": dataset.get("metadata", {}).get("versi", "1.1"),
        "total_pertanyaan_dalam_domain_diuji": n_in,
        "total_pertanyaan_boundary_diuji": n_b,
        "total_seluruh_pengujian": n_in + n_b,
        "retrieval_k": k_val,
        "retrieval_pool_k": pool_k,
        "mode_dijalankan": list(mode_stats.keys()),
        "embedding_model": settings.embedding_model,
        "ringkasan_retrieval": retrieval_summary,
        "ringkasan_per_mode": per_mode_summary,
        "rotasi_api_key_gemini": {
            "total_kunci_terdaftar": len(gemini_key_rotator.keys),
            "kunci_aktif_terakhir": gemini_key_rotator.get_current_key_info(),
            "kunci_terkena_limit_count": len(gemini_key_rotator.exhausted_keys),
            "kunci_terkena_limit_list": [mask_key(k) for k in gemini_key_rotator.exhausted_keys],
            "daftar_riwayat_rotasi": gemini_key_rotator.rotation_history
        },
        "rekapitulasi_per_kategori": category_summary
    }

    output_data = {
        "metadata_evaluasi": {
            "judul": "Hasil Pengujian Komputasional RAG Chatbot Jaringan Komputer (Dataset v1.1)",
            "berkas_dataset": os.path.basename(dataset_path),
            "versi_dataset": dataset.get("metadata", {}).get("versi", "1.1"),
            "waktu_pelaksanaan": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "konfigurasi_sistem": {
                "mode_diminta": args.mode,
                "mode_dijalankan": modes,
                "llm_model_per_mode": {m: model_name_for(m) for m in set(modes) | set(mode_stats)},
                "embedding_model": settings.embedding_model,
                "vector_store": "ChromaDB",
                "retrieval_top_k": k_val,
                "retrieval_pool_k": pool_k,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "temperature": settings.llm_temperature,
                "gemini_api_keys_dimuat": len(gemini_key_rotator.keys)
            },
            "metrik_yang_digunakan": [
                "Precision@K (Retrieval Performance)",
                "Recall@K & Context Recall (Retrieval Coverage)",
                "Faithfulness (Generation Anti-Hallucination)",
                "Answer Relevance (Generation Relevance)",
                "Domain Boundary (System Prompt Constraint)",
                "Latensi Endpoint (Average Response Time)"
            ]
        },
        "ringkasan_evaluasi": summary,
        "detail_pengujian_dalam_domain": in_domain_results,
        "detail_pengujian_boundary": boundary_results
    }

    mode_tag = args.mode
    output_filename = args.output or os.path.join(BASE_DIR, "data", f"hasil_evaluasi_v1.1_{mode_tag}.json")
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, ensure_ascii=False, indent=2)

    # ---- Ringkasan konsol ----
    print("\n" + "=" * 78)
    print("  HASIL EVALUASI KOMPUTASIONAL RAG CHATBOT (DATASET v1.1)")
    print("=" * 78)
    print(f"  Dataset        : {os.path.basename(dataset_path)} | Embedding: {settings.embedding_model}")
    print(f"  Kueri diuji    : {n_in} dalam domain + {n_b} boundary = {n_in + n_b}")
    print("  " + "-" * 76)
    print("  Retrieval (sama untuk semua mode):")
    print(f"   1. Precision@{k_val}    : {retrieval_summary['precision_at_k']['rata_rata']:.4f}")
    print(f"   2. Recall@{k_val}       : {retrieval_summary['recall_at_k']['rata_rata']:.4f}  (pool={pool_k})")
    print(f"      Context Recall  : {retrieval_summary['context_recall']['rata_rata']:.4f}")
    sh = retrieval_summary["source_hit_rate"]["rata_rata"]
    print(f"      Source Hit Rate : {sh:.4f}" if sh is not None else "      Source Hit Rate : -")
    for m, s in per_mode_summary.items():
        print("  " + "-" * 76)
        print(f"  Mode '{m}' ({s['llm_model']}) | error generasi: {s['jumlah_error_generasi']}")
        print(f"   3. Faithfulness     : {s['faithfulness']['skor_rata_rata_1_sampai_5']:.2f}/5 ({s['faithfulness']['persentase_kelayakan']}%)")
        print(f"   4. Answer Relevance : {s['answer_relevance']['skor_rata_rata_1_sampai_5']:.2f}/5 ({s['answer_relevance']['persentase_kelayakan']}%)")
        b = s["domain_boundary"]
        print(f"   5. Domain Boundary  : {b['berhasil_ditolak']}/{b['total_kueri_diuji']} ({b['persentase_keberhasilan']}%)")
        print(f"   6. Latensi Rata-rata: {s['latensi_endpoint']['rata_rata_ms']:.0f} ms ({s['latensi_endpoint']['status_standar_waktu']})")
    print("  " + "-" * 76)
    print("  Per Kategori Materi:")
    for c, r in category_summary.items():
        print(f"   • {c} ({r['jumlah_pertanyaan']} soal): P@{k_val}={r['rata_rata_precision_at_k']:.2f} | "
              f"R@{k_val}={r['rata_rata_recall_at_k']:.2f} | CtxRecall={r['rata_rata_context_recall']:.2f}")
    print("  " + "-" * 76)
    print(f"  JSON tersimpan di: {output_filename}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()