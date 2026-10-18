"""Stage 1 - Ingestion: read the PDF and save clean text, one record per page."""
import json
import re
import unicodedata

import pymupdf

from src.config import RAW_PDF, PAGES_JSONL

# This sentence is printed at the top of every page of this PDF.
# If we keep it, every chunk contains the same words, which weakens search.
PAGE_HEADER = "Artificial intelligence: opportunities and implications for the future of decision making"


def clean_text(text: str) -> str:
    """Fix the usual problems of text copied out of a PDF."""
    text = text.replace(PAGE_HEADER, " ")             # remove the repeated page header
    text = unicodedata.normalize("NFKC", text)        # "fi" symbol -> f + i
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # rejoin hyphenated line breaks
    text = re.sub(r"\s*\n\s*", " ", text)             # line breaks -> spaces
    return re.sub(r"\s{2,}", " ", text).strip()       # collapse repeated spaces

def repair_garbled(raw: str) -> str:
    """Some pages store letters shifted by 29. Shift them back if the page looks garbled."""
    if len(raw) < 200 or raw.lower().split().count("the") > 0:
        return raw  # page is normal, leave it alone
    fixed = "".join(chr(ord(c) + 29) if not c.isspace() else c for c in raw)
    if fixed.lower().split().count("the") < 3:
        return raw  # shifting did not help, keep the original
    for bad, good in (("ϙ", '"'), ("Ó", '"'), ("Î", "-")):
        fixed = fixed.replace(bad, good)
    return fixed

def extract_pages(pdf_path):
    """Return a list with one dictionary per page: source, page number, text."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            pages.append({
                "source": pdf_path.name,
                "page": number,
                "text": repair_garbled(clean_text(page.get_text())),
            })
    return pages


def main():
    pages = extract_pages(RAW_PDF)
    pages = [p for p in pages if p["page"] not in (1, 3, 21)]
    PAGES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(PAGES_JSONL, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    words = sum(len(p["text"].split()) for p in pages)
    print(f"Source: {RAW_PDF.name}")
    print(f"Pages: {len(pages)} | Words: {words}")
    print(f"Saved to: {PAGES_JSONL}")


if __name__ == "__main__":
    main()
