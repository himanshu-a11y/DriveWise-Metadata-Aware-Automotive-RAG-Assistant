"""
pipeline.py
-------------------------------
The "conductor" - wires chunker -> retriever -> reranker -> generator ->
logger together in the exact order described in problem statement
section 9. Shared by both the notebook and app_web.py so they can never
drift apart into two different implementations.

Error handling
--------------
Each pipeline stage is wrapped in its own try/except block.  On failure
it raises a PipelineError carrying the stage name, so callers (app.py,
app_web.py) can show a meaningful message without exposing a raw traceback
to the user.
"""

from .chunker import load_all_brochures
from .retriever import VectorStore
from .reranker import rerank
from .generator import generate_answer
from .logger import log_query, Timer
from .validator import PipelineError, fuzzy_match_cars


class DriveWisePipeline:
    def __init__(self, brochure_folder: str = "data/brochures"):
        try:
            self.chunks = load_all_brochures(brochure_folder)
        except Exception as e:
            raise PipelineError("initialisation", e) from e

        try:
            self.store = VectorStore(self.chunks)
        except Exception as e:
            raise PipelineError("initialisation", e) from e

    def list_available_cars(self):
        seen = set()
        cars = []
        for c in self.chunks:
            key = (c["brand"], c["model"])
            if key not in seen:
                seen.add(key)
                cars.append(key)
        return cars

    def fuzzy_search_cars(self, query: str, top_n: int = 3) -> list:
        """
        Return up to *top_n* cars whose name best matches *query*.

        Each result is a dict with keys: brand, model, label, score.
        Delegates to validator.fuzzy_match_cars so the logic lives in one place.
        """
        cars = self.list_available_cars()
        return fuzzy_match_cars(query, cars, top_n=top_n)

    def ask(self, brand: str, model: str, query: str,
            retrieve_k: int = 5, final_k: int = 1,
            min_rerank_score: float = -10.0) -> dict:
        """
        Run the full RAG pipeline for a validated query.

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

        Raises
        ------
        PipelineError
            If any individual stage (retrieval / reranking / generation /
            logging) raises an unexpected exception.
        """
        with Timer() as t:
            # ── Stage 1: Retrieval ──────────────────────────────────────────
            try:
                retrieved = self.store.search(
                    query, brand=brand, model=model, top_k=retrieve_k
                )
            except Exception as e:
                raise PipelineError("retrieval", e) from e

            # ── Stage 2: Re-ranking ─────────────────────────────────────────
            try:
                context_chunks = rerank(query, retrieved, top_n=final_k)
                # Drop chunks the cross-encoder itself rates as irrelevant
                context_chunks = [
                    r for r in context_chunks
                    if r.get("rerank_score", 0) >= min_rerank_score
                ]
            except Exception as e:
                raise PipelineError("reranking", e) from e

            # ── Stage 3: Generation ─────────────────────────────────────────
            try:
                result = generate_answer(query, context_chunks)
            except Exception as e:
                raise PipelineError("generation", e) from e

        # ── Stage 4: Logging ────────────────────────────────────────────────
        failed = len(result["sources"]) == 0
        try:
            log_query(
                brand=brand, model=model, query=query,
                sources=result["sources"],
                response_time_seconds=t.elapsed,
                failed=failed,
            )
        except Exception as e:
            # Logging failure is non-fatal — warn but don't crash the user's session
            raise PipelineError("logging", e) from e

        result["response_time_seconds"] = round(t.elapsed, 4)
        result["context_chunks"] = context_chunks  # used by evaluator.py
        return result
