"""
chunker.py (primary pipeline)
-----------------------------
Reads brochure .txt files and splits them into structured chunks, using
the [SECTION: ...] and [PAGE: ...] markers written into each file.

Each chunk carries the 5 metadata fields the project brief asks for:
    brand, model, section, page_number, version
plus the source file name and a unique chunk_id used for source
attribution later (so an answer can point back to an exact page).
"""

import os
import re


def parse_brochure(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    filename = os.path.basename(path)
    brand = _match(r"BRAND:\s*(.+)", raw)
    model = _match(r"MODEL:\s*(.+)", raw)
    version = _match(r"VERSION:\s*(.+)", raw)

    parts = re.split(r"\[SECTION:\s*(.+?)\]", raw)
    chunks = []
    for i in range(1, len(parts), 2):
        section_name = parts[i].strip().lower()
        section_body = parts[i + 1].strip()

        page_match = re.match(r"\[PAGE:\s*(\d+)\]\s*", section_body)
        if page_match:
            page_number = int(page_match.group(1))
            section_text = section_body[page_match.end():].strip()
        else:
            page_number = 0
            section_text = section_body

        if not section_text:
            continue

        chunks.append({
            "text": section_text,
            "brand": brand,
            "model": model,
            "section": section_name,
            "version": version,
            "source_file": filename,
            "page_number": page_number,
            "chunk_id": f"{filename}::{section_name}::p{page_number}",
        })
    return chunks


def _match(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else "Unknown"


def load_all_brochures(folder: str) -> list[dict]:
    all_chunks = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            all_chunks.extend(parse_brochure(os.path.join(folder, fname)))
    return all_chunks
