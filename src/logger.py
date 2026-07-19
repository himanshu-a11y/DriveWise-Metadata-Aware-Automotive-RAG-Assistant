"""
logger.py
---------
The problem statement asks for logging and monitoring: every time a user
asks a question, we should record what happened so we can debug problems
later and measure how well the system is performing.

We keep it simple: every query becomes ONE line of JSON appended to
logs/query_log.jsonl. This file format is called "JSON Lines" - each
line is its own independent JSON object, which makes it easy to read
the log line by line without loading the whole file into memory.

Each log entry stores:
    - timestamp
    - brand / model selected
    - the question asked
    - retrieval_results: exactly which chunks were retrieved (chunk
      reference, section, page number, relevance score) - this is the
      'retrieval results' field the problem statement asks for
    - how many chunks were retrieved
    - response time in seconds
    - answer_generation_status: "success" or "failed_no_relevant_data"
    - whether the query failed (no relevant chunks found) or succeeded
"""

import json
import os
import time
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "query_log.jsonl")


def log_query(brand: str, model: str, query: str, sources: list,
              response_time_seconds: float, failed: bool):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "brand": brand,
        "model": model,
        "query": query,
        "retrieval_results": [
            {"chunk_reference": s["chunk_reference"], "section": s["section"],
             "page_number": s["page_number"], "relevance_score": s["relevance_score"]}
            for s in sources
        ],
        "num_chunks_found": len(sources),
        "response_time_seconds": round(response_time_seconds, 4),
        "answer_generation_status": "failed_no_relevant_data" if failed else "success",
        "failed": failed,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


class Timer:
    """Small helper so pipeline.py can measure response time with a 'with' block."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
