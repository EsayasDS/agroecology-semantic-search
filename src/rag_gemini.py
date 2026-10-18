"""Stage 5b - RAG with Gemini: retrieve passages, the LLM answers only from them."""
import csv
import os
import time

from google import genai

from src.answer import QUESTIONS, THRESHOLD, LIVE_WORDS, load

MODEL = "gemini-3.8-flash"

PROMPT = """You answer questions using ONLY the passages below.
Rules:
- Use only facts that are written in the passages.
- After each fact, write the page in brackets, like [page 6].
- If the passages do not answer the question, reply exactly: NOT IN DOCUMENT
- Keep the answer to 3 sentences or fewer.

PASSAGES:
{passages}

QUESTION: {question}
ANSWER:"""


def retrieve(question, embedder, collection):
    qv = embedder.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=qv, n_results=3,
                           include=["documents", "metadatas", "distances"])
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def main():
    embedder, collection = load()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rows = []
    for q in QUESTIONS:
        if any(w in q.lower() for w in LIVE_WORDS):
            hits = []
            status, dist, pages, text = "REFUSED", "n/a", "", "Needs live information."
        else:
            hits = retrieve(q, embedder, collection)
            dist = hits[0][2]
            pages = ",".join(str(h[1].get("page", "?")) for h in hits)
            if dist > THRESHOLD:
                status, text = "REFUSED", "No passage close enough."
            else:
                passages = "\n\n".join(f"[page {h[1].get('page', '?')}] {h[0]}" for h in hits)
                try:
                    resp = client.models.generate_content(
                        model=MODEL, contents=PROMPT.format(passages=passages, question=q))
                    text = (resp.text or "").strip()
                except Exception as e:
                    text = f"ERROR: {e}"
                status = "REFUSED_BY_LLM" if "NOT IN DOCUMENT" in text else "ANSWERED"
                time.sleep(4)
            dist = f"{dist:.3f}"
        rows.append([q, status, dist, pages, text])
        print("=" * 80)
        print(f"QUESTION: {q}")
        print(f"[{status}] distance {dist} | pages {pages}")
        print(text)
    with open("eval/rag_gemini_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question", "status", "distance", "pages", "answer", "assessment"])
        w.writerows([r + [""] for r in rows])
    print("\nSaved: eval/rag_gemini_log.csv")


if __name__ == "__main__":
    main()
