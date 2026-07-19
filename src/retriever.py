"""
retriever.py
--------------------------------
Turns brochure chunks into real neural embeddings (using the
sentence-transformers library) and stores them in a FAISS vector
database for fast similarity search. This matches the problem
statement's tech stack: "Embedding model for semantic search" +
"Vector database for brochure embeddings".

"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_embedder = None


def get_embedder():
    """Load the embedding model once and reuse it (loading is slow)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


class VectorStore:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)

    def search(self, query: str, brand: str = None, model: str = None, top_k: int = 5):
        """
        Step 1: embed the query
        Step 2: search the WHOLE FAISS index (so metadata filtering never
                 accidentally starves us of candidates)
        Step 3: keep only results matching the selected brand/model
        Step 4: return the top_k closest matches
        FAISS returns L2 *distance*, where smaller = more similar, so we
        convert it to a similarity score (smaller distance -> higher score)
        for consistency with the rest of the pipeline.
        """
        embedder = get_embedder()
        query_vector = embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k=len(self.chunks))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if brand and chunk["brand"].lower() != brand.lower():
                continue
            if model and chunk["model"].lower() != model.lower():
                continue
            similarity = 1.0 / (1.0 + float(dist))  # convert distance -> 0..1 similarity
            results.append({"chunk": chunk, "score": similarity})
            if len(results) >= top_k:
                break
        return results
