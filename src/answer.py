"""Stage 5 - Answer step (extractive): best sentences + page number, or refuse."""
import csv
import re
import chromadb
from sentence_transformers import SentenceTransformer, util

from src.config import CHROMA_DIR, EMBED_MODEL, COLLECTION

THRESHOLD = 0.62  # above this distance the evidence is too weak
LIVE_WORDS = ("current", "currently", "today", "right now", "latest", "price")

QUESTIONS = [
    "What is machine learning?",
    "What is deep learning?",
    "What is supervised learning?",
    "How can AI make public services more efficient?",
    "Who is responsible when an AI system makes a wrong decision?",
    "Why is transparency important when AI is used to make decisions?",
    "How can bias get into an AI system?",
    "What are the ethical questions raised by artificial intelligence?",
    "How could AI change jobs and work?",
    "What is the current price of Nvidia stock today?",
    "How do I train a transformer in PyTorch?",
    "What is the capital of France?",
    "What is the exact accuracy of ChatGPT on medical exams?",
]

CHECKS = {
    "What is machine learning?": "weak: shows terminology, not the definition",
    "What is deep learning?": "supported",
    "What is supervised learning?": "weak: first sentence right, second is padding",
    "How can AI make public services more efficient?": "supported",
    "Who is responsible when an AI system makes a wrong decision?": "supported",
    "Why is transparency important when AI is used to make decisions?": "supported",
    "How can bias get into an AI system?": "supported",
    "What are the ethical questions raised by artificial intelligence?": "weak: shows seminar history, not the questions",
    "How could AI change jobs and work?": "supported",
}


def load():
    model = SentenceTransformer(EMBED_MODEL)
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    return model, collection


def answer(question, model, collection):
    if any(w in question.lower() for w in LIVE_WORDS):
        return "REFUSED", None, "The question needs live information. A static report cannot answer it."
    qv = model.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=qv, n_results=3,
                           include=["documents", "metadatas", "distances"])
    distance = res["distances"][0][0]
    if distance > THRESHOLD:
        return "REFUSED", distance, "The report has no passage close enough to answer this."
    text = res["documents"][0][0]
    page = res["metadatas"][0][0].get("page", "?")
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 6] or [text]
    scores = util.cos_sim(
        model.encode(question, convert_to_tensor=True, normalize_embeddings=True),
        model.encode(sents, convert_to_tensor=True, normalize_embeddings=True),
    )[0]
    top = sorted(scores.argsort(descending=True)[:2].tolist())
    return "ANSWERED", distance, " ".join(sents[i] for i in top) + f"  [page {page}]"


def main():
    model, collection = load()
    rows = []
    for q in QUESTIONS:
        status, dist, text = answer(q, model, collection)
        d = f"{dist:.3f}" if dist is not None else "n/a"
        check = CHECKS.get(q, "correct refusal" if status == "REFUSED" else "")
        rows.append([q, status, d, check, text])
        print("=" * 80)
        print(f"QUESTION: {q}")
        print(f"[{status}] distance {d}  ->  {check}")
        print(text)
    with open("eval/answer_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question", "status", "distance", "assessment", "answer"])
        w.writerows(rows)
    answered = sum(r[1] == "ANSWERED" for r in rows)
    print(f"\nAnswered: {answered} | Refused: {len(rows) - answered}")
    print("Saved: eval/answer_log.csv")


if __name__ == "__main__":
    main()
