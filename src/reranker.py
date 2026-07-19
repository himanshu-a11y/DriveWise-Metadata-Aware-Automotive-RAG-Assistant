"""
reranker.py
-------------------------------
Uses a real Cross-Encoder model to re-score the chunks that the FAISS
search already retrieved. A cross-encoder reads the query and a chunk
TOGETHER (rather than comparing separately-computed vectors), which
makes it slower but much more accurate at judging true relevance. This
is a standard "retrieve fast, then re-rank carefully" pattern.

'top_n' also performs Context Window Control: only the very best chunk(s)
are kept, so the language model isn't given noisy, irrelevant text.
"""

from sentence_transformers import CrossEncoder

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, retrieved: list[dict], top_n: int = 1) -> list[dict]:
    if not retrieved:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [[query, r["chunk"]["text"]] for r in retrieved]
    scores = cross_encoder.predict(pairs)

    for r, score in zip(retrieved, scores):
        r["rerank_score"] = float(score)

    reranked = sorted(retrieved, key=lambda r: r["rerank_score"], reverse=True)
    return reranked[:top_n]
