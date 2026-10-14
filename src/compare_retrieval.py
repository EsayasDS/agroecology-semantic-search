
from pathlib import Path
import json

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "results" / "chunks.json"
DB_PATH = ROOT / "results" / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

QUESTIONS = [
    "What is agroecology?",
    "How does agroecology combine traditional and industrial agriculture?",
    "What are the benefits of crop diversification?",
    "How can farmers improve soil fertility?",
    "What challenges can farmers face when adopting agroecological practices?",
    "What is the current price of wheat in Ethiopia today?",
]


def main():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    documents = [chunk["text"] for chunk in chunks]

    # Baseline: lexical retrieval using TF-IDF.
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Semantic retrieval: load the embedding model only once.
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("agroecology_manual")

    print("\nRETRIEVAL COMPARISON")
    print(f"Documents: {len(documents)}")
    print(f"Questions: {len(QUESTIONS)}")

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("QUESTION:", question)

        # TF-IDF top result.
        query_vector = vectorizer.transform([question])
        lexical_scores = cosine_similarity(
            query_vector, tfidf_matrix
        ).flatten()
        lexical_index = lexical_scores.argmax()

        lexical_chunk = chunks[lexical_index]

        print("\n[TF-IDF TOP RESULT]")
        print("PDF page:", lexical_chunk["pdf_page"])
        print("Score:", round(float(lexical_scores[lexical_index]), 4))
        print(lexical_chunk["text"][:900])

        # Semantic search top result.
        query_embedding = model.encode(
            [question], normalize_embeddings=True
        ).tolist()

        semantic = collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        print("\n[SEMANTIC SEARCH TOP RESULT]")
        print("PDF page:", semantic["metadatas"][0][0]["pdf_page"])
        print("Cosine distance:",
              round(float(semantic["distances"][0][0]), 4))
        print(semantic["documents"][0][0][:900])


if __name__ == "__main__":
    main()
