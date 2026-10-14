\# Agroecology Semantic Search



A Python-based document retrieval prototype that compares TF-IDF keyword retrieval with semantic search over an agroecology training manual.



\## Features



\- Extract PDF text and page metadata with PyMuPDF.

\- Split text into overlapping chunks.

\- Generate text embeddings using Sentence Transformers.

\- Store and retrieve embeddings with ChromaDB.

\- Compare semantic retrieval against a TF-IDF baseline.

\- Display retrieved passages, source pages, and cosine distances.

\- Demonstrate a basic warning for selected current-information questions.



\## Technology Stack



\- Python 3.11

\- PyMuPDF

\- Sentence Transformers (`all-MiniLM-L6-v2`)

\- ChromaDB

\- scikit-learn



\## Project Structure



```text

agroecology-rag/

├── data/                  # Source PDF; excluded from Git

├── results/               # Experiment log and generated outputs

├── src/

│   ├── ingestion.py

│   ├── chunking.py

│   ├── semantic\_search.py

│   ├── compare\_retrieval.py

│   ├── rag\_demo.py

│   └── test\_search.py

├── report/

│   └── seminar\_report.md

├── .gitignore

└── requirements.txt

```



\## Run the Project



Create and activate a Python virtual environment, then install the dependencies:



```bash

pip install -r requirements.txt

```



Place the source PDF in the `data/` directory and run the scripts from the project root in the appropriate order:



```bash

python src/ingestion.py

python src/chunking.py

python src/semantic\_search.py

python src/compare\_retrieval.py

python src/rag\_demo.py

```



\## Evaluation



The initial experiment compares TF-IDF and semantic retrieval across six questions. Observations show that semantic search can retrieve useful passages for some questions but may also return incomplete or irrelevant evidence.



The prototype retrieves evidence rather than generating complete answers with a language model. Its current-information warning uses simple rules and is not a general answerability detector.



\## Limitations



\- Small evaluation set.

\- Qualitative relevance judgements.

\- Cosine distance is not accuracy or answer confidence.

\- Static documents cannot establish changing facts such as current market prices.

\- PDF extraction may not preserve all tables, figures, or layout.



\## Academic Use



This project is an exploratory academic prototype. Retrieved evidence should be checked against the original source before drawing conclusions.



