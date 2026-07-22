"""
pipeline.py
-------------------------------
The "conductor" - wires chunker -> retriever -> reranker -> generator ->
logger together in the exact order described in problem statement
section 9. Shared by both the notebook and app_web.py so they can never
drift apart into two different implementations.
"""

from .chunker import load_all_brochures
from .retriever import VectorStore
from .reranker import rerank
from .generator import generate_answer
from .logger import log_query, Timer


class DriveWisePipeline:
    def __init__(self, brochure_folder: str = "data/brochures"):
        self.chunks = load_all_brochures(brochure_folder)
        self.store = VectorStore(self.chunks)

    def list_available_cars(self):
        seen = set()
        cars = []
        for c in self.chunks:
            key = (c["brand"], c["model"])
            if key not in seen:
                seen.add(key)
                cars.append(key)
        return cars

    def ask(self, brand: str, model: str, query: str,
            retrieve_k: int = 5, final_k: int = 1,
            min_rerank_score: float = -10.0) -> dict:
        """
        min_rerank_score: the cross-encoder's re-rank score below which we
        treat the match as "not actually relevant" rather than force an
        answer. The cross-encoder used here (ms-marco-MiniLM-L-6-v2)
        generally scores genuinely relevant question/passage pairs above
        0, and unrelated pairs below 0 - but this can vary. BEFORE your
        viva, calibrate it: run one clearly in-scope question (e.g. "what
        is the mileage?") and one clearly out-of-scope question (e.g.
        "what is the price?") through this pipeline, print
        context_chunks[0]['rerank_score'] for both, and set this
        threshold somewhere between the two scores you see.
        """
        with Timer() as t:
            retrieved = self.store.search(query, brand=brand, model=model, top_k=retrieve_k)
            context_chunks = rerank(query, retrieved, top_n=final_k)
            # drop the match if the cross-encoder itself doesn't think
            # it's relevant, instead of forcing a confident-looking answer
            context_chunks = [r for r in context_chunks if r.get("rerank_score", 0) >= min_rerank_score]
            result = generate_answer(query, context_chunks)

        failed = len(result["sources"]) == 0
        log_query(
            brand=brand, model=model, query=query,
            sources=result["sources"],
            response_time_seconds=t.elapsed,
            failed=failed,
        )

        result["response_time_seconds"] = round(t.elapsed, 4)
        result["context_chunks"] = context_chunks  # used by evaluator.py
        return result
