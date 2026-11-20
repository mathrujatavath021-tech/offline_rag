import os
import time
import json
import hashlib
import threading
import numpy as np
from typing import Generator, Dict, Any, Optional, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

import rag

# Global singletons
_engine_lock = threading.RLock()
_llm_instance: Optional[Llama] = None
_embedder_instance: Optional[SentenceTransformer] = None

# Active cancellation flags: request_id -> threading.Event
active_cancellations: Dict[str, threading.Event] = {}

# Query & Embedding Caches
# query_hash -> {"vector": np.ndarray}
_query_embed_cache: Dict[str, np.ndarray] = {}
MAX_EMBED_CACHE_SIZE = 500

# (corpus_hash, query_hash, lang) -> {"answer": str, "sources": list, "tok_count": int}
_generation_cache: Dict[Tuple[str, str, str], dict] = {}
MAX_GEN_CACHE_SIZE = 200

# Corpus version hash (updated whenever chunks change)
_corpus_version: str = "v0"


def invalidate_cache():
    """Invalidates generation cache when documents are modified or re-indexed."""
    global _corpus_version, _generation_cache
    with _engine_lock:
        _corpus_version = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        _generation_cache.clear()
        print(f"[*] RAG Cache invalidated. New corpus version: {_corpus_version}")


def get_embedder() -> SentenceTransformer:
    global _embedder_instance
    with _engine_lock:
        if _embedder_instance is None:
            t0 = time.perf_counter()
            _embedder_instance = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
            print(f"[*] Embedder loaded in {time.perf_counter() - t0:.2f}s")
        return _embedder_instance


def get_llm() -> Optional[Llama]:
    global _llm_instance
    with _engine_lock:
        if _llm_instance is None and os.path.exists(rag.MODEL_PATH):
            t0 = time.perf_counter()
            # Physical cores = 4 on this system, n_batch=512 for fast context prefill
            _llm_instance = Llama(
                model_path=rag.MODEL_PATH,
                n_ctx=2048,
                n_threads=4,
                n_threads_batch=8,
                n_batch=512,
                verbose=False
            )
            print(f"[*] Mistral 7B LLM pre-warmed in {time.perf_counter() - t0:.2f}s (Threads=4, Batch=512)")
        return _llm_instance


def get_query_embedding(query: str) -> np.ndarray:
    """Returns query embedding with in-memory LRU cache."""
    q_key = query.strip().lower()
    if q_key in _query_embed_cache:
        return _query_embed_cache[q_key]

    embedder = get_embedder()
    vec = embedder.encode([query])
    if len(_query_embed_cache) > MAX_EMBED_CACHE_SIZE:
        _query_embed_cache.pop(next(iter(_query_embed_cache)))
    _query_embed_cache[q_key] = vec
    return vec


def cancel_request(request_id: str):
    """Signals ongoing generation for request_id to abort immediately."""
    if request_id in active_cancellations:
        active_cancellations[request_id].set()


class RAGEngine:
    @staticmethod
    def retrieve_context(query: str, chunks: list, embeddings: np.ndarray, top_k: int = 3) -> Tuple[str, List[str], float]:
        """
        Retrieves top-k relevant document chunks with deduplication.
        Returns (context_text, source_names, retrieval_latency_ms).
        """
        t0 = time.perf_counter()
        if not chunks or embeddings is None or len(embeddings) == 0:
            return "No local documents indexed.", [], 0.0

        q_vec = get_query_embedding(query)
        scores = cosine_similarity(q_vec, embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        matched_texts = []
        sources = []
        for idx in top_indices:
            if scores[idx] > 0.22:
                matched_texts.append(chunks[idx]["text"])
                src = chunks[idx].get("source", "doc")
                if src not in sources:
                    sources.append(src)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        if matched_texts:
            return "\n\n".join(matched_texts)[:1500], sources, latency_ms
        return "No sufficiently relevant local documents found.", [], latency_ms

    @staticmethod
    def stream_answer(
        query: str,
        context_text: str,
        matched_sources: List[str],
        selected_lang: str = "English",
        request_id: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Token-by-token streaming generator with cancellation support and latency diagnostics.
        """
        cancel_event = threading.Event()
        if request_id:
            active_cancellations[request_id] = cancel_event

        t_start = time.perf_counter()
        t_first_token = None
        full_answer = ""
        tok_count = 0

        # Check generation cache
        cache_key = (_corpus_version, query.strip().lower(), selected_lang)
        if cache_key in _generation_cache:
            cached = _generation_cache[cache_key]
            # Yield cached answer word by word smoothly
            words = cached["answer"].split(" ")
            for w in words:
                yield {"token": w + " ", "cached": True}
                time.sleep(0.01)
            yield {
                "done": True,
                "answer": cached["answer"],
                "sources": cached["sources"],
                "cached": True,
                "diagnostics": {
                    "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
                    "ttft_ms": 5.0,
                    "tok_per_sec": 120.0,
                    "token_count": len(words)
                }
            }
            if request_id:
                active_cancellations.pop(request_id, None)
            return

        wants_diagram = any(x in query.lower() for x in ["flowchart", "diagram"])
        current_time_str = time.strftime("%A, %B %d, %Y %I:%M %p")

        if wants_diagram:
            prompt = f"""[INST] Draw a vertical ASCII flowchart for: "{query}".
Use [ ] for boxes and v for arrows. No markdown code blocks.
Context: {context_text} [/INST]"""
        else:
            prompt = f"""[INST] System: {current_time_str}.
Context: {context_text}
Question: {query}
If context is missing, answer using your own knowledge. [/INST]"""

        llm = get_llm()
        if not llm:
            yield {"token": "Error: Local LLM model not found."}
            yield {"done": True, "answer": "Error: Local LLM model not found.", "sources": []}
            if request_id:
                active_cancellations.pop(request_id, None)
            return

        try:
            stream = llm(prompt, max_tokens=512, stop=["</s>"], stream=True)
            for chunk in stream:
                if cancel_event.is_set():
                    yield {"token": "\n[Generation stopped by user]"}
                    break

                if t_first_token is None:
                    t_first_token = time.perf_counter()

                tok = chunk["choices"][0]["text"]
                full_answer += tok
                tok_count += 1
                yield {"token": tok}

        except Exception as e:
            yield {"token": f"\nError during generation: {e}"}

        finally:
            if request_id:
                active_cancellations.pop(request_id, None)

        t_end = time.perf_counter()
        total_gen_time = t_end - t_start
        ttft_s = (t_first_token - t_start) if t_first_token else 0
        tok_per_sec = round(tok_count / (t_end - (t_first_token or t_start)), 2) if (t_end - (t_first_token or t_start)) > 0 else 0

        clean_answer = full_answer.strip()

        # Save to generation cache if valid and not cancelled
        if not cancel_event.is_set() and len(clean_answer) > 20:
            if len(_generation_cache) > MAX_GEN_CACHE_SIZE:
                _generation_cache.pop(next(iter(_generation_cache)))
            _generation_cache[cache_key] = {
                "answer": clean_answer,
                "sources": matched_sources,
                "tok_count": tok_count
            }

        yield {
            "done": True,
            "answer": clean_answer,
            "sources": matched_sources,
            "diagnostics": {
                "total_ms": round(total_gen_time * 1000, 1),
                "ttft_ms": round(ttft_s * 1000, 1),
                "tok_per_sec": tok_per_sec,
                "token_count": tok_count,
                "cached": False
            }
        }
