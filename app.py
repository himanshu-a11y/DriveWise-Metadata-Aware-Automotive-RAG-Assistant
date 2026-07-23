"""
app.py
------
Command line interface for the primary DriveWise pipeline (real
embeddings + FAISS + cross-encoder re-ranking + flan-t5 generation).

First run will download ~250MB of models from HuggingFace, so an
internet connection is required the first time.

Run with: python app.py

Improvements (evaluation feedback)
-----------------------------------
- Input validation: empty / too-short / symbol-only queries are rejected
  and the user is re-prompted instead of crashing.
- Fuzzy car search: typing a name (e.g. "creta", "nexon") instead of a
  number shows the top-3 matching vehicles and lets the user confirm.
- Error handling: PipelineError from any stage is caught and shown as a
  clean one-line message — no raw Python traceback exposed to the user.
"""

import sys

from src.pipeline import DriveWisePipeline
from src.validator import PipelineError, validate_query, fuzzy_match_cars


# ── Helpers ─────────────────────────────────────────────────────────────────

def _print_car_list(cars: list) -> None:
    print("\nAvailable cars in the brochure database:")
    for i, (brand, model) in enumerate(cars, start=1):
        print(f"  {i:>2}. {brand} {model}")


def _select_car_numeric(choice: str, cars: list):
    """Try to parse *choice* as a 1-based index. Returns (brand, model) or None."""
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(cars):
            return cars[idx]
    except ValueError:
        pass
    return None


def _select_car_fuzzy(query: str, cars: list):
    """
    Use fuzzy matching to find the best car for a text query.
    Shows the top-3 candidates and asks the user to confirm.
    Returns (brand, model) on confirmation, or None if cancelled.
    """
    matches = fuzzy_match_cars(query, cars, top_n=3)
    if not matches:
        print("  No matching cars found.")
        return None

    print(f"\n  Did you mean one of these? (searched for: '{query}')")
    for i, m in enumerate(matches, start=1):
        print(f"    {i}. {m['label']}  (match score: {m['score']:.0%})")
    print("    0. None of these — go back")

    confirm = input("\n  Enter number to confirm (or 0 to go back): ").strip()
    try:
        ci = int(confirm) - 1
        if 0 <= ci < len(matches):
            m = matches[ci]
            return m["brand"], m["model"]
    except ValueError:
        pass
    return None


def _ask_car_selection(cars: list):
    """
    Prompt the user to select a car.

    Accepts:
      - A number (1-based index shown in the list)
      - A text search term (brand name, model name, or partial match)

    Loops until a valid selection is made or the user types 'quit'.
    """
    _print_car_list(cars)
    print("\nTip: enter a number to pick directly, or type a name to search "
          "(e.g. 'creta', 'nexon', 'tata'). Type 'quit' to exit.")

    while True:
        choice = input("\nSelect a car: ").strip()

        if choice.lower() in ("quit", "exit", "q"):
            return None

        if not choice:
            print("  Please enter a number or a car name.")
            continue

        # ── Try numeric first ──────────────────────────────────────────────
        result = _select_car_numeric(choice, cars)
        if result:
            return result

        # ── Fall back to fuzzy text search ─────────────────────────────────
        result = _select_car_fuzzy(choice, cars)
        if result:
            return result

        # User cancelled fuzzy or no match — show list again
        _print_car_list(cars)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" DriveWise - Metadata Aware Automotive RAG Assistant")
    print("=" * 60)
    print("Loading models (this can take a minute the first time)...")

    # ── Pipeline initialisation ───────────────────────────────────────────
    try:
        pipeline = DriveWisePipeline()
    except PipelineError as e:
        print(f"\n[ERROR] Failed during {e.stage}: {e.cause}")
        print("Check that the 'data/brochures' folder exists and is readable.")
        sys.exit(1)

    cars = pipeline.list_available_cars()
    if not cars:
        print("\n[ERROR] No brochures found in data/brochures/. "
              "Add at least one .pdf or .txt brochure and try again.")
        sys.exit(1)

    # ── Car selection ─────────────────────────────────────────────────────
    selection = _ask_car_selection(cars)
    if selection is None:
        print("Goodbye!")
        return

    brand, model = selection
    print(f"\nYou selected: {brand} {model}")
    print("Ask anything about this car. Type 'exit' or 'quit' to leave.\n")

    # ── Q&A loop ──────────────────────────────────────────────────────────
    while True:
        raw_query = input("You: ").strip()

        if raw_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # ── Input validation ───────────────────────────────────────────
        try:
            query = validate_query(raw_query)
        except ValueError as ve:
            print(f"  ⚠  {ve}\n")
            continue

        # ── Pipeline call ──────────────────────────────────────────────
        try:
            result = pipeline.ask(brand, model, query)
        except PipelineError as e:
            print(f"\n  [ERROR] Pipeline failed at '{e.stage}' stage: {e.cause}")
            print("  Please try rephrasing your question or restart the app.\n")
            continue

        # ── Display answer ─────────────────────────────────────────────
        print(f"\nDriveWise: {result['answer']}\n")
        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(
                    f"  - {s['brand']} {s['model']} | section: {s['section']} "
                    f"| page: {s['page_number']} | file: {s['brochure_file']} "
                    f"| chunk ref: {s['chunk_reference']} | relevance: {s['relevance_score']}"
                )
        print(f"(answered in {result['response_time_seconds']}s)\n")


if __name__ == "__main__":
    main()
