
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer

# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "results" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "agroecology_manual"

TOP_K = 3
MAX_DISTANCE = 0.55


# ==================================================
# LOAD MODEL AND DATABASE
# ==================================================

print("Loading semantic search model...")
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(name=COLLECTION_NAME)

print(f"Loaded {collection.count()} indexed chunks.")


# ==================================================
# RETRIEVAL
# ==================================================

def retrieve(question, top_k=TOP_K):
    """Retrieve passages with their PDF page numbers."""

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

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append({
            "text": document,
            "page": metadata.get("pdf_page", "Unknown"),
            "distance": float(distance),
        })

    return passages


# ==================================================
# BASIC EVIDENCE CHECK
# ==================================================

def check_evidence(question, passages):
    """
    Apply simple heuristic checks.

    These checks are not a trained answerability classifier
    and cannot guarantee that a question is answerable.
    """

    if not passages:
        return False, "No passages were retrieved."

    # Reject results that are too distant according to this
    # initial heuristic. Validate the threshold experimentally.
    if passages[0]["distance"] > MAX_DISTANCE:
        return False, "The retrieved passages are not sufficiently similar."

    # Detect explicit requests for changing, current information
    # that this static manual cannot reliably establish.
    current_info_patterns = [
        r"\b(today|right now|currently|latest|current price)\b",
        r"\bthis week\b",
        r"\bcurrent exchange rate\b",
    ]

    if any(
        re.search(pattern, question.lower())
        for pattern in current_info_patterns
    ):
        return False, (
            "This question requests current information. "
            "The static manual cannot establish today's value."
        )

    # A close semantic match is not enough to prove that
    # the passage directly answers the question.
    # Display the passages for human inspection.
    return True, (
        "Relevant passages were retrieved, but direct support "
        "must be checked by reading them."
    )


# ==================================================
# DISPLAY RESULTS
# ==================================================

def answer_question(question):
    passages = retrieve(question)
    supported_by_heuristic, explanation = check_evidence(
        question, passages
    )

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    if not supported_by_heuristic:
        print("\nINSUFFICIENT EVIDENCE WARNING")
        print(explanation)

        print(
            "\nThe passages below are shown for transparency; "
            "they should not be treated as an answer."
        )
    else:
        print("\nRETRIEVED EVIDENCE")
        print(explanation)

    for i, passage in enumerate(passages, start=1):
        print(f"\nPassage {i} | PDF page {passage['page']}")
        print(f"Cosine distance: {passage['distance']:.4f}")
        print(passage["text"])
        print("-" * 70)

    print("\nLIMITATION")
    print(
        "This demo retrieves passages; it does not generate "
        "a verified answer. Similarity is not proof of support."
    )


# ==================================================
# INTERACTIVE LOOP
# ==================================================

def main():
    print("\nAGROECOLOGY MANUAL — SEMANTIC SEARCH DEMO")
    print("Ask a question about the PDF.")
    print("Type 'exit' to stop.")

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
            print(f"\nError while processing the question: {error}")


if __name__ == "__main__":
    main()
