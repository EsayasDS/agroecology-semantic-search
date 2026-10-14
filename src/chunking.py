
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion import extract_pdf

PDF_PATH = PROJECT_ROOT / "data" / "Agroecology-Manual.pdf"
OUTPUT_PATH = PROJECT_ROOT / "results" / "chunks.txt"


def split_into_chunks(text, chunk_size=180, overlap=35):
    """Split text into overlapping passages of words."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def main():
    pages = extract_pdf(PDF_PATH)
    all_chunks = []

    for page in pages:
        passages = split_into_chunks(page["text"])

        for number, passage in enumerate(passages, start=1):
            all_chunks.append({
                "source": page["source"],
                "pdf_page": page["pdf_page"],
                "printed_page": page.get("printed_page"),
                "chunk_number": number,
                "text": passage,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(f"PDF page: {chunk['pdf_page']}\n")
            file.write(f"Printed page: {chunk['printed_page']}\n")
            file.write(f"Chunk: {chunk['chunk_number']}\n")
            file.write(chunk["text"] + "\n")
            file.write("=" * 80 + "\n")

    print("\nCHUNKING REPORT")
    print(f"Pages processed: {len(pages)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
