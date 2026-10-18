"""Compare keyword search (TF-IDF) with semantic search on the same questions."""
import csv

from src.config import ROOT
from src.retrieve import Retriever

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

OUT = ROOT / "eval" / "retrieval_comparison.csv"


def main():
    r = Retriever()
    rows = []
    for q in QUESTIONS:
        kw, sem = r.keyword(q), r.semantic(q)
        print("=" * 80)
        print("QUESTION:", q)
        print(f"\n[KEYWORD  ] page {kw[0]['page']}  score {kw[0]['score']:.3f}")
        print("   ", kw[0]["text"][:220], "...")
        print(f"[SEMANTIC ] page {sem[0]['page']}  distance {sem[0]['distance']:.3f}")
        print("   ", sem[0]["text"][:220], "...")
        rows.append([q, "keyword", [h["page"] for h in kw], round(kw[0]["score"], 3), kw[0]["text"][:200], "", ""])
        rows.append([q, "semantic", [h["page"] for h in sem], round(sem[0]["distance"], 3), sem[0]["text"][:200], "", ""])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question", "method", "top3_pages", "top1_score_or_distance",
                    "top1_snippet", "assessment", "notes"])
        w.writerows(rows)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
