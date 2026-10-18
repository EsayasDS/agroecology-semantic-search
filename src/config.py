from pathlib import Path

# Project root = the folder that contains "src"
ROOT = Path(__file__).resolve().parents[1]

# Input and stage 1 output
RAW_PDF = ROOT / "data" / "raw" / "AI-GOScience.pdf"
PAGES_JSONL = ROOT / "data" / "processed" / "pages.jsonl"

# Stage 2 and 3 outputs
CHUNKS_JSONL = ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "processed" / "chroma"

# Embedding model and database collection
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "ai_report"

