
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


def extract_pdf(pdf_path: str) -> list[dict]:
    """Extract text page by page while preserving source page numbers."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {path.resolve()}\n"
            "Check the file path and try again."
        )

    pages = []

    with fitz.open(path) as document:
        total_pages = len(document)

        for index, page in enumerate(document):
            pdf_page_number = index + 1
            text = page.get_text("text")

            # Remove excessive whitespace.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            # Remove a simple printed footer such as "Page | 9".
            text = re.sub(
                r"(?im)^\s*Page\s*\|\s*\d+\s*$",
                "",
                text
            ).strip()

            if not text:
                continue

            # Detect a printed page label when present.
            printed_page_match = re.search(
                r"(?im)^\s*Page\s*\|\s*(\d+)\s*$",
                page.get_text("text")
            )

            printed_page_number = (
                int(printed_page_match.group(1))
                if printed_page_match
                else None
            )

            pages.append({
                "source": path.name,
                "pdf_page": pdf_page_number,
                "printed_page": printed_page_number,
                "text": text,
                "word_count": len(text.split()),
            })

    if total_pages == 0:
        raise ValueError("The PDF contains no pages.")

    pages_with_text = len(pages)

    print("=" * 60)
    print("PDF INGESTION REPORT")
    print("=" * 60)
    print(f"File: {path.name}")
    print(f"Total PDF pages: {total_pages}")
    print(f"Pages containing text: {pages_with_text}")
    print(f"Total extracted words: {sum(p['word_count'] for p in pages)}")

    if pages_with_text < total_pages / 2:
        print("\nWARNING: Many pages contain no extractable text.")
        print("The PDF may require OCR.")

    if pages:
        print("\nFirst extracted page:")
        print(f"PDF page: {pages[0]['pdf_page']}")
        print(f"Printed page: {pages[0]['printed_page']}")
        print(f"Words: {pages[0]['word_count']}")
        print("-" * 60)
        print(pages[0]["text"][:1200])

    return pages


if __name__ == "__main__":
    default_path = "data/Agroecology-Manual.pdf"
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    try:
        extracted_pages = extract_pdf(pdf_path)
        print(f"\nSUCCESS: Extracted {len(extracted_pages)} pages.")
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
