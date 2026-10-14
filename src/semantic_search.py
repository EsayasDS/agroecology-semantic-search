
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "results" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "agroecology_manual"

TOP_K = 3

# This is a rough filtering threshold, not proof of answerability.
MAX_DISTANCE = 0.55


# ==================================================
# LOAD MODEL AND DATABASE
# ==================================================

print("Loading semantic search model...")

model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(DB_PATH))

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception as error:
    raise RuntimeError(
        "Could not find the indexed collection. "
        "Run: python src\\semantic_search.py"
    ) from error

if collection.count() == 0:
    raise RuntimeError(
        "The search index is empty. "
        "Run: python src\\semantic_search.py"
    )

print(f"Index ready: {collection.count()} chunks loaded.")


# ==================================================
# RETRIEVE RELEVANT PASSAGES
# ==================================================

def retrieve(question, top_k=TOP_K):
    """Retrieve passages and their stored source metadata."""

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    passages = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        pdf_page = metadata.get("pdf_page", "Unknown")
        printed_page = metadata.get("printed_page", -1)

        # The cover page may not have a printed page number.
        if printed_page == -1:
            printed_page = None

        passages.append({
            "text": document,
            "pdf_page": pdf_page,
            "printed_page": printed_page,
            "chunk_number": metadata.get("chunk_number", "Unknown"),
            "source": metadata.get("source", "Agroecology Manual"),
            "distance": float(distance),
        })

    return passages


# ==================================================
# DISPLAY RETRIEVED EVIDENCE
# ==================================================

def answer_question(question):
    """
    Display retrieved evidence.

    This version does not generate a new answer using an LLM.
    It shows passages from the PDF for the user to inspect.
    """

    passages = retrieve(question)

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    if not passages:
        print("\nNo passages were retrieved.")
        return

    # A low distance does not guarantee that evidence answers
    # the question, so inspect the passage content as well.
    if passages[0]["distance"] > MAX_DISTANCE:
        print("\nThe retrieved passages may not contain enough evidence.")
        print("Try a more specific question or inspect the PDF directly.")

    else:
        print("\nRETRIEVED EVIDENCE")
        print("-" * 70)

        for rank, passage in enumerate(passages, start=1):
            print(f"\nPassage {rank}")

            print(f"Source: {passage['source']}")
            print(f"PDF page: {passage['pdf_page']}")

            if passage["printed_page"] is not None:
                print(f"Printed page: {passage['printed_page']}")

            print(f"Chunk: {passage['chunk_number']}")
            print(f"Cosine distance: {passage['distance']:.4f}")
            print("\nPassage text:")
            print(passage["text"])
            print("-" * 70)

    print("\nEVALUATION NOTE")
    print(
        "Retrieved passages are evidence, not proof that the "
        "question is fully answered."
    )
    print(
        "Verify the source page and do not assume the document "
        "contains information it does not provide."
    )


# ==================================================
# INTERACTIVE DEMO
# ==================================================

def main():
    print("\nAGROECOLOGY MANUAL — SEMANTIC SEARCH DEMO")
    print("Ask a question about the indexed PDF.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDemo finished.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Demo finished.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer_question(question)
        except Exception as error:
            print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()
