"""
chunker.py
-----------------------------
Reads brochure files and splits them into structured chunks.

Supports two formats:
  1. PDF files (.pdf)  - real manufacturer brochures downloaded from
     official websites (e.g. hyundaimotorindia.com, tatamotors.com).
     Text is extracted page-by-page using PyMuPDF, then each page is
     assigned to a section using keyword detection.

  2. Structured .txt files - legacy format with [SECTION:] / [PAGE:]
     markers, kept for backward compatibility.

Each chunk carries the 5 metadata fields the project brief asks for:
    brand, model, section, page_number, version
plus the source file name and a unique chunk_id used for source
attribution later (so an answer can point back to an exact page).
"""

import os
import re

# ---------------------------------------------------------------------------
# Section keywords: used to classify each PDF page into a logical section
# ---------------------------------------------------------------------------
SECTION_KEYWORDS = {
    "engine and performance": [
        "engine", "horsepower", "torque", "transmission", "rpm", "cylinder",
        "displacement", "bhp", "turbo", "gearbox", "drivetrain", "powertrain",
    ],
    "mileage and fuel efficiency": [
        "mileage", "fuel", "kmpl", "efficiency", "range", "consumption",
        "arai", "petrol", "diesel", "cng", "electric", "battery",
    ],
    "safety": [
        "airbag", "abs", "ebd", "safety", "brake", "isofix", "seatbelt",
        "ncap", "esc", "traction", "collision", "pedestrian", "camera",
    ],
    "dimensions": [
        "length", "width", "height", "wheelbase", "ground clearance",
        "boot", "luggage", "kerb weight", "turning radius", "mm", "litre",
    ],
    "interior and comfort": [
        "sunroof", "seat", "leather", "air condition", "climate", "infotainment",
        "touchscreen", "steering", "comfort", "ventilat", "ambient", "interior",
    ],
    "exterior and design": [
        "design", "colour", "color", "alloy", "wheel", "headlamp", "led",
        "grille", "bumper", "roof", "spoiler", "exterior", "body",
    ],
    "infotainment and connectivity": [
        "android auto", "apple carplay", "bluetooth", "wifi", "navigation",
        "speaker", "usb", "wireless", "connected", "app", "voice",
    ],
    "warranty and service": [
        "warranty", "service", "maintenance", "guarantee", "year", "km",
        "roadside", "assistance",
    ],
}


def _detect_section(text: str) -> str:
    """Return the best-matching section name for a block of text."""
    text_lower = text.lower()
    best_section = "general"
    best_count = 0
    for section, keywords in SECTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_section = section
    return best_section


def _brand_model_from_filename(filename: str):
    """
    Infer brand and model from filename convention:
      hyundai_creta.pdf  ->  brand='Hyundai',  model='Creta'
      tata_nexon.pdf     ->  brand='Tata',     model='Nexon'
      maruti_baleno.pdf  ->  brand='Maruti Suzuki', model='Baleno'
    """
    stem = os.path.splitext(filename)[0].lower()
    brand_map = {
        "hyundai": "Hyundai",
        "tata":    "Tata",
        "maruti":  "Maruti Suzuki",
        "honda":   "Honda",
        "toyota":  "Toyota",
        "kia":     "Kia",
        "mahindra": "Mahindra",
    }
    brand = "Unknown"
    model = "Unknown"
    for key, val in brand_map.items():
        if stem.startswith(key):
            brand = val
            rest = stem[len(key):].lstrip("_- ")
            model = rest.replace("_", " ").title()
            break
    return brand, model


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------
def parse_pdf_brochure(path: str) -> list[dict]:
    """Extract chunks from a real PDF brochure using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required to read PDF brochures. "
            "Install it with: pip install PyMuPDF"
        )

    filename = os.path.basename(path)
    brand, model = _brand_model_from_filename(filename)

    doc = fitz.open(path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < 30:          # skip nearly-blank / image-only pages
            continue

        section = _detect_section(text)
        chunks.append({
            "text": text,
            "brand": brand,
            "model": model,
            "section": section,
            "version": "official",
            "source_file": filename,
            "page_number": page_num,
            "chunk_id": f"{filename}::{section}::p{page_num}",
        })

    doc.close()
    return chunks


# ---------------------------------------------------------------------------
# Legacy .txt parser (kept for backward compatibility)
# ---------------------------------------------------------------------------
def parse_brochure(path: str) -> list[dict]:
    """Parse a structured .txt brochure file with [SECTION:] / [PAGE:] tags."""
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


# ---------------------------------------------------------------------------
# Main loader — picks PDF or .txt automatically
# ---------------------------------------------------------------------------
def load_all_brochures(folder: str) -> list[dict]:
    """
    Load all brochures from folder.
    - .pdf files are parsed with PyMuPDF (real manufacturer brochures)
    - .txt files are parsed with the legacy structured parser
    If both a .pdf and a .txt exist for the same car, the PDF takes priority.
    """
    all_chunks = []
    pdf_stems = set()

    # Pass 1: load PDFs first
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(folder, fname)
            try:
                chunks = parse_pdf_brochure(fpath)
                all_chunks.extend(chunks)
                pdf_stems.add(os.path.splitext(fname)[0].lower())
                print(f"[chunker] Loaded PDF: {fname}  ({len(chunks)} chunks)")
            except Exception as e:
                print(f"[chunker] WARNING: Could not parse PDF {fname}: {e}")

    # Pass 2: load .txt only if no PDF exists for that car
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".txt"):
            stem = os.path.splitext(fname)[0].lower()
            if stem in pdf_stems:
                print(f"[chunker] Skipping {fname} (PDF version loaded)")
                continue
            fpath = os.path.join(folder, fname)
            chunks = parse_brochure(fpath)
            all_chunks.extend(chunks)
            print(f"[chunker] Loaded TXT: {fname}  ({len(chunks)} chunks)")

    return all_chunks
