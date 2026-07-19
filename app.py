"""
app.py
------
Command line interface for the primary DriveWise pipeline (real
embeddings + FAISS + cross-encoder re-ranking + flan-t5 generation).

First run will download ~250MB of models from HuggingFace, so an
internet connection is required the first time.

Run with: python3 app.py
"""

from src.pipeline import DriveWisePipeline


def main():
    print("=" * 60)
    print(" DriveWise - Metadata Aware Automotive RAG Assistant")
    print("=" * 60)
    print("Loading models (this can take a minute the first time)...")

    pipeline = DriveWisePipeline()
    cars = pipeline.list_available_cars()

    print("\nAvailable cars in the brochure database:")
    for i, (brand, model) in enumerate(cars, start=1):
        print(f"  {i}. {brand} {model}")

    choice = input("\nSelect a car by number: ").strip()
    try:
        brand, model = cars[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Exiting.")
        return

    print(f"\nYou selected: {brand} {model}")
    print("Ask anything about this car. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        result = pipeline.ask(brand, model, query)

        print(f"\nDriveWise: {result['answer']}\n")
        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(f"  - {s['brand']} {s['model']} | section: {s['section']} "
                      f"| page: {s['page_number']} | file: {s['brochure_file']} "
                      f"| chunk ref: {s['chunk_reference']} | relevance: {s['relevance_score']}")
        print(f"(answered in {result['response_time_seconds']}s)\n")


if __name__ == "__main__":
    main()
