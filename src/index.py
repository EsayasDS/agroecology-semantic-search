"""Stage 3 - Embedding and indexing: turn chunks into vectors and store them in ChromaDB."""
import json

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_JSONL, CHROMA_DIR, EMBED_MODEL, COLLECTION


def load_chunks():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    # 1. Embed: one 384-number vector per chunk
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # 2. Store: vectors + text + page number in ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)   # start clean on every run
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=vectors.tolist(),
        metadatas=[
            {"source": c["source"], "page": c["page"], "chunk": c["chunk"]}
            for c in chunks
        ],
    )
    print(f"Embedding size: {vectors.shape[1]} numbers per chunk")
    print(f"Chunks indexed: {collection.count()}")
    return model, collection


def smoke_test(model, collection, question="What is overfitting?"):
    """Embed one question and show the 3 closest chunks."""
    query = model.encode([question], normalize_embeddings=True)
    result = collection.query(query_embeddings=query.tolist(), n_results=3)
    print(f"\nTest question: {question}")
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        print(f"\n[page {meta['page']}] distance {dist:.3f}")
        print(doc[:300] + "...")


if __name__ == "__main__":
    model, collection = build_index()
    smoke_test(model, collection)
