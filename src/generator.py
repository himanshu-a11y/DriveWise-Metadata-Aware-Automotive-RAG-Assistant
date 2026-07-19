"""
generator.py
--------------------------------
Takes the best re-ranked chunk(s) and asks a real local language model
(google/flan-t5-small) to write a natural-sounding answer, using ONLY
that retrieved text as context. This matches the problem statement's
"Language model for answer generation".

flan-t5-small is used (not a larger model) because it's small enough to
run on a normal laptop CPU without a paid API key.

Source attribution: every answer also returns exactly which brand,
model, section, page number, file, and chunk reference it came from -
this is what lets a user (or examiner) verify the answer against the
real brochure.
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_model = None
_tokenizer = None


def get_generation_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        _model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return _model, _tokenizer


def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    if not context_chunks:
        return {
            "answer": "I couldn't find anything about that in the brochure for this car. "
                      "Try rephrasing your question or check you selected the right model.",
            "sources": [],
        }

    model, tokenizer = get_generation_model()

    context_text = " ".join(r["chunk"]["text"] for r in context_chunks)
    prompt = (
        f"Answer the question based on the context.\n"
        f"Context: {context_text}\nQuestion: {query}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=150)
    answer_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sources = []
    for r in context_chunks:
        c = r["chunk"]
        sources.append({
            "brand": c["brand"],
            "model": c["model"],
            "section": c["section"],
            "page_number": c["page_number"],
            "brochure_file": c["source_file"],
            "chunk_reference": c["chunk_id"],
            "relevance_score": round(r.get("rerank_score", r.get("score", 0.0)), 3),
        })

    return {"answer": answer_text, "sources": sources}
