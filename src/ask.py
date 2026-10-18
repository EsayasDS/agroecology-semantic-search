"""Interactive demo: keyword vs semantic retrieval, refusal rules,
extractive answer, Gemini answer, and a citation check."""
import csv
import os
import re
import textwrap

from google import genai

from src.answer import THRESHOLD, LIVE_WORDS, answer
from src.rag_gemini import MODEL, PROMPT, retrieve
from src.retrieve import Retriever

LOG = "eval/manual_log.csv"


def snippet(text, n=150):
    return " ".join(text.split())[:n] + "..."


def load_saved():
    try:
        with open("eval/rag_gemini_log.csv", encoding="utf-8-sig") as f:
            return {r["question"]: r["answer"] for r in csv.DictReader(f)}
    except FileNotFoundError:
        return {}


def log_row(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["question", "status", "distance", "semantic_pages",
                        "extractive_answer", "gemini_answer", "citation_check"])
        w.writerow(row)


def show_page(r, cmd):
    parts = cmd.split()
    if len(parts) != 2 or not parts[1].isdigit():
        print("Use: /page 15")
        return
    n = parts[1]
    chunks = sorted((c for c in r.chunks if str(c["page"]) == n), key=lambda c: c["chunk"])
    if not chunks:
        print("No chunks for that page (pages 1, 3 and 21 were dropped).")
        return
    for c in chunks:
        print(f"\n--- page {n}, chunk {c['chunk']} ---")
        print(textwrap.fill(c["text"], 90))


def main():
    r = Retriever()
    saved = load_saved()
    key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key) if key else None
    print("Ask a question about the UK AI report. Type /page 15 to read a page. "
          "Empty line quits.")
    print("Gemini: LIVE" if client else
          "Gemini: no key set. Saved answers will be shown for the 13 test questions.")

    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            break
        if q.lower().startswith("/page"):
            show_page(r, q)
            continue
        print("=" * 80)

        print("KEYWORD (TF-IDF), top 3:")
        for h in r.keyword(q):
            print(f"  page {h['page']:>2}  score    {h['score']:.3f}  {snippet(h['text'])}")

        sem = r.semantic(q)
        print("\nSEMANTIC (embeddings), top 3:")
        for h in sem:
            print(f"  page {h['page']:>2}  distance {h['distance']:.3f}  {snippet(h['text'])}")

        live = [w for w in LIVE_WORDS if w in q.lower()]
        top = sem[0]["distance"]
        print("\nREFUSAL CHECK")
        print(f"  live-information words found: {live or 'none'}")
        print(f"  closest distance: {top:.3f} (refuse above {THRESHOLD})")

        status, dist, text = answer(q, r.model, r.collection)
        print(f"\nEXTRACTIVE ANSWER [{status}]")
        print(textwrap.fill(text, 90, initial_indent="  ", subsequent_indent="  "))

        gem, check = "(skipped)", ""
        if client is None:
            gem = ("[SAVED RUN, not live] " + saved[q]) if q in saved else "(no API key: skipped)"
        elif live or top > THRESHOLD:
            gem = "(not called: the refusal rules stopped the question first)"
        else:
            hits = retrieve(q, r.model, r.collection)
            print("\nPASSAGES SENT TO GEMINI")
            for doc, meta, d in hits:
                print(f"  [page {meta.get('page', '?')}] distance {d:.3f}  {snippet(doc, 100)}")
            passages = "\n\n".join(f"[page {m.get('page', '?')}] {d}" for d, m, _ in hits)
            try:
                resp = client.models.generate_content(
                    model=MODEL, contents=PROMPT.format(passages=passages, question=q))
                gem = (resp.text or "").strip()
            except Exception as e:
                gem = f"ERROR: {e}"
            if "NOT IN DOCUMENT" not in gem and not gem.startswith("ERROR"):
                retrieved = {str(m.get("page", "?")) for _, m, _ in hits}
                cited = set(re.findall(r"\d+", " ".join(re.findall(r"\[pages?[^\]]*\]", gem))))
                ok = bool(cited) and cited <= retrieved
                check = (f"cited pages {sorted(cited)} | retrieved pages {sorted(retrieved)} | "
                         f"all cited pages were retrieved: {'YES' if ok else 'NO'}")

        print("\nGEMINI ANSWER")
        print(textwrap.fill(gem, 90, initial_indent="  ", subsequent_indent="  "))
        if check:
            print(f"\nCITATION CHECK\n  {check}")

        log_row([q, status, f"{top:.3f}", ",".join(str(h["page"]) for h in sem), text, gem, check])


if __name__ == "__main__":
    main()