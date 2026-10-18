"""Stage 2 - Chunking: split each page into small, sentence-aligned chunks."""
import json
import re

from src.config import PAGES_JSONL, CHUNKS_JSONL

MAX_WORDS = 120          # target chunk size in words
OVERLAP_SENTENCES = 1    # sentences repeated between neighbouring chunks


def split_sentences(text):
    """Split after . ! ? when the next sentence starts with a capital, digit or bracket."""
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)


def chunk_sentences(sentences, max_words=MAX_WORDS, overlap=OVERLAP_SENTENCES):
    """Group sentences until a chunk reaches max_words, then start the next one."""
    chunks, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if sum(len(s.split()) for s in current) >= max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
    if current and (not chunks or len(current) > overlap):
        chunks.append(" ".join(current))
    return chunks


def main():
    with open(PAGES_JSONL, encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]

    records = []
    for page in pages:
        sentences = [s for s in split_sentences(page["text"]) if s.strip()]
        for number, text in enumerate(chunk_sentences(sentences), start=1):
            records.append({
                "id": f"p{page['page']}-c{number}",
                "source": page["source"],
                "page": page["page"],
                "chunk": number,
                "text": text,
            })

    CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sizes = [len(r["text"].split()) for r in records]
    print(f"Chunks: {len(records)}")
    print(f"Words per chunk: min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)}")
    print(f"Saved to: {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
