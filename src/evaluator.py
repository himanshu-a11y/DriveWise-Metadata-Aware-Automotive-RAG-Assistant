"""
evaluator.py (primary pipeline)
--------------------------------
Implements all 3 evaluation dimensions the problem statement asks for
in section 7 (the earlier notebook version only had 2 of these):

1. Answer Correctness - does the answer contain the fact we expected?
2. Faithfulness - is the answer's wording actually grounded in the
   retrieved brochure text (not invented by the LLM)?
3. Context Relevance - did retrieval actually find the right chunk?
"""

TEST_CASES = [
    {"brand": "Hyundai", "model": "Creta", "query": "What is the mileage of the diesel variant?",
     "expected_keyword": "20.2", "expected_section": "mileage and fuel efficiency"},
    {"brand": "Tata", "model": "Nexon", "query": "How many airbags does it have?",
     "expected_keyword": "airbag", "expected_section": "safety"},
    {"brand": "Maruti Suzuki", "model": "Baleno", "query": "What is the boot space?",
     "expected_keyword": "318", "expected_section": "dimensions"},
    {"brand": "Hyundai", "model": "Creta", "query": "Does it have a sunroof?",
     "expected_keyword": "sunroof", "expected_section": "interior and comfort"},
    {"brand": "Tata", "model": "Nexon", "query": "What is the ground clearance?",
     "expected_keyword": "209", "expected_section": "dimensions"},
]


def answer_correctness(answer: str, expected_keyword: str) -> bool:
    return expected_keyword.lower() in answer.lower()


def faithfulness(answer: str, context_chunks: list[dict]) -> float:
    """What fraction of the answer's words also appear in the retrieved
    context. High score = grounded answer, not invented by the LLM."""
    if not context_chunks:
        return 0.0
    context_words = set(" ".join(r["chunk"]["text"].lower() for r in context_chunks).split())
    answer_words = [w.strip(".,") for w in answer.lower().split()]
    if not answer_words:
        return 0.0
    grounded = sum(1 for w in answer_words if w in context_words)
    return round(grounded / len(answer_words), 2)


def context_relevance(context_chunks: list[dict], expected_section: str) -> float:
    """Did we retrieve the section that actually answers this question?"""
    if not context_chunks:
        return 0.0
    sections = [r["chunk"]["section"] for r in context_chunks]
    return 1.0 if expected_section in sections else 0.0


def run_evaluation(ask_fn):
    """ask_fn(brand, model, query) -> {"answer":..., "sources":..., "context_chunks":...}"""
    print(f"{'Query':45} {'Correct?':10} {'Faithful':10} {'Ctx.Rel.':10}")
    print("-" * 80)
    correct_count = 0
    for case in TEST_CASES:
        result = ask_fn(case["brand"], case["model"], case["query"])
        answer = result["answer"]
        context_chunks = result.get("context_chunks", [])

        is_correct = answer_correctness(answer, case["expected_keyword"])
        correct_count += int(is_correct)
        f_score = faithfulness(answer, context_chunks)
        c_score = context_relevance(context_chunks, case["expected_section"])

        print(f"{case['query'][:43]:45} {str(is_correct):10} {f_score:<10} {c_score:<10}")

    accuracy = round(correct_count / len(TEST_CASES), 2)
    print("-" * 80)
    print(f"Overall Answer Correctness: {accuracy * 100:.0f}% ({correct_count}/{len(TEST_CASES)})")
