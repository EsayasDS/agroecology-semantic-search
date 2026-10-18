\# Cited Document Q\&A with RAG



A small Retrieval-Augmented Generation (RAG) demo that answers questions from a public PDF, cites the page, and \*\*refuses when the evidence is weak\*\*.



Built for the DaSc4162 Student Seminar Challenge (Group 4: Embeddings, Vector Databases and RAG), Debre Berhan University.



\*\*Document:\*\* UK Government Office for Science, \*Artificial intelligence: opportunities and implications for the future of decision making\* (2016), used under the Open Government Licence.



\## How it works



PDF -> clean text per page -> sentence-aligned chunks (\~120 words, 1-sentence overlap) -> embeddings -> ChromaDB -> top-3 retrieval -> refusal check -> answer + page number



| Part | Choice |

|---|---|

| Chunks | 76 |

| Embedding model | `all-MiniLM-L6-v2` (384 dimensions) |

| Vector database | ChromaDB (local) |

| Keyword baseline | TF-IDF (scikit-learn) |

| Refusal rules | Distance above 0.62, or live-information words |

| Answer step | Extractive (no LLM) and Gemini (`gemini-3.8-flash`) |



\## Results (13 questions: 9 answerable, 4 not)



\- \*\*Search:\*\* semantic search was better than keyword search on 5 questions, tied on 4, and was never worse.

\- \*\*Extractive answers:\*\* 6 supported, 3 weak, 4 correct refusals.

\- \*\*Gemini answers:\*\* 7 supported, 2 wrong refusals, 4 correct refusals, 0 invented facts.



Full logs are in `eval/`.



\## Run it



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt



python -m src.ingest      # read and clean the PDF

python -m src.chunk       # split into chunks

python -m src.index       # embed and store in ChromaDB

python -m src.compare     # keyword vs semantic search

python -m src.answer      # extractive answers with refusal



$env:GEMINI\_API\_KEY = "your-key-here"

python -m src.rag\_gemini  # Gemini answers from retrieved passages

```



The API key is read from an environment variable and is never stored in this repository.



\## Structure



```

data/raw/        source PDF

data/processed/  cleaned pages and chunks

src/             pipeline code (one file per stage)

eval/            experiment logs (CSV)

```

