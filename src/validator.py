"""
validator.py
------------
Centralised input validation and error types for DriveWise.

Imported by pipeline.py, app.py, and app_web.py so that validation
rules are defined exactly once.

Contents
--------
PipelineError   – custom exception that carries the pipeline stage name
                  (retrieval / reranking / generation / logging) so the
                  UI can give the user a meaningful message instead of a
                  raw Python traceback.

validate_query  – sanitise a free-text question before it enters the
                  pipeline.  Raises ValueError with a human-readable
                  message on failure; returns the cleaned string on success.

fuzzy_match_cars – rank the available cars by similarity to a text query
                   using Python's built-in difflib (no extra dependencies).
                   Returns the top-N (brand, model) tuples sorted by score.
"""

import difflib
import re


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """
    Raised when a named stage of the DriveWise pipeline fails.

    Attributes
    ----------
    stage : str
        Which stage failed — one of:
        'initialisation', 'retrieval', 'reranking', 'generation', 'logging'
    cause : Exception
        The original exception that triggered this error.
    """

    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(f"[{stage}] {type(cause).__name__}: {cause}")


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------

# Minimum number of *word* characters required in a query
_MIN_WORD_CHARS = 3

# A query is rejected if it contains only non-word characters (symbols/digits)
_ONLY_SYMBOLS = re.compile(r"^[\W\d_]+$", re.UNICODE)


def validate_query(query: str) -> str:
    """
    Validate and clean a user query before it enters the pipeline.

    Rules
    -----
    1. Strip leading/trailing whitespace.
    2. Reject empty or whitespace-only strings.
    3. Reject strings that are purely numeric or purely symbolic
       (e.g. "123", "???", "!!!").
    4. Reject strings with fewer than 3 word-characters
       (e.g. a single letter or a two-character abbreviation).

    Returns
    -------
    str
        The cleaned query string.

    Raises
    ------
    ValueError
        With a descriptive, user-facing message explaining why the
        query was rejected.
    """
    cleaned = query.strip()

    if not cleaned:
        raise ValueError("Query cannot be empty. Please type a question about the car.")

    if _ONLY_SYMBOLS.match(cleaned):
        raise ValueError(
            "Query contains only symbols or numbers. "
            "Please ask a proper question, e.g. 'What is the mileage?'"
        )

    word_chars = re.sub(r"[\W\d_]", "", cleaned, flags=re.UNICODE)
    if len(word_chars) < _MIN_WORD_CHARS:
        raise ValueError(
            f"Query is too short (need at least {_MIN_WORD_CHARS} letters). "
            "Please be more specific."
        )

    return cleaned


# ---------------------------------------------------------------------------
# Fuzzy car search
# ---------------------------------------------------------------------------

def fuzzy_match_cars(
    query: str,
    cars: list,
    top_n: int = 3,
    cutoff: float = 0.0,
) -> list:
    """
    Rank *cars* by how closely their label matches *query*.

    Parameters
    ----------
    query : str
        The text the user typed (partial brand, model, or both).
    cars : list of (brand, model) tuples
        The full catalogue returned by pipeline.list_available_cars().
    top_n : int
        Maximum number of matches to return.
    cutoff : float
        Minimum similarity ratio (0.0–1.0) to include a result.
        Default 0.0 means always return at least one result unless
        the list is empty.

    Returns
    -------
    list of dict
        Each dict has keys: 'brand', 'model', 'label', 'score'.
        Sorted descending by score.
    """
    if not cars:
        return []

    q = query.strip().lower()

    scored = []
    for brand, model in cars:
        label = f"{brand} {model}"
        label_lower = label.lower()

        # SequenceMatcher gives a ratio in [0, 1]
        ratio = difflib.SequenceMatcher(None, q, label_lower).ratio()

        # Boost if the query is a substring of the label (partial match)
        if q in label_lower:
            ratio = max(ratio, 0.6)

        # Boost even more if any single word in the query matches a word in label
        q_words = q.split()
        label_words = label_lower.split()
        for qw in q_words:
            if any(lw.startswith(qw) for lw in label_words):
                ratio = max(ratio, 0.5)

        scored.append({
            "brand": brand,
            "model": model,
            "label": label,
            "score": round(ratio, 3),
        })

    # Sort by score descending, then alphabetically for ties
    scored.sort(key=lambda x: (-x["score"], x["label"]))

    # Apply cutoff and limit
    results = [s for s in scored if s["score"] >= cutoff]
    return results[:top_n]
