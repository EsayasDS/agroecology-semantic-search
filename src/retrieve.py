"""Stage 4 - Retrieval: semantic search (embeddings) and keyword search (TF-IDF)."""
import json

import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CHUNKS_JSONL, CHROMA_DIR, EMBED_MODEL, COLLECTION


class Retriever:
    def __init__(self):
        with open(CHUNKS_JSONL, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        # Semantic side: embedding model + the ChromaDB collection we built
        self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION)

        # Keyword side: TF-IDF matrix built from the same chunk texts
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.tfidf.fit_transform([c["text"] for c in self.chunks])

    def semantic(self, question, k=3):
        """Closest chunks by meaning. 'distance': smaller = better."""
        q = self.model.encode([question], normalize_embeddings=True)
        res = self.collection.query(query_embeddings=q.tolist(), n_results=k)
        return [
            {"id": i, "page": m["page"], "distance": d, "text": t}
            for i, m, d, t in zip(
                res["ids"][0], res["metadatas"][0],
                res["distances"][0], res["documents"][0],
            )
        ]

    def keyword(self, question, k=3):
        """Best chunks by shared words. 'score': larger = better."""
        q = self.tfidf.transform([question])
        scores = cosine_similarity(q, self.matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [
            {"id": self.chunks[i]["id"], "page": self.chunks[i]["page"],
             "score": float(scores[i]), "text": self.chunks[i]["text"]}
            for i in top
        ]
