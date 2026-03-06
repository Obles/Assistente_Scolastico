# scripts/common_config.py
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Determina la ROOT del progetto in modo assoluto, NON relativo alla current working dir
# La ROOT è sempre: freud-kb/
CURRENT_DIR = Path(__file__).resolve().parent        # freud-kb/scripts
PROJECT_DIR = CURRENT_DIR.parent                     # freud-kb/

load_dotenv(PROJECT_DIR / ".env")  # carica SEMPRE il .env nella root corretta

# Costruisci i path SEMPRE rispetto alla root del progetto
BUILD_DIR = PROJECT_DIR / os.getenv("BUILD_DIR", "build")
DATA_DIR  = PROJECT_DIR / os.getenv("DATA_DIR",  "data")
PAGES_DIR = BUILD_DIR / os.getenv("PAGES_DIR", "pages")
DOCUMENTS_DIR = PROJECT_DIR / os.getenv("DOCUMENTS_DIR", "data/documents")

# Output JSON scraper
CATALOGO_FILENAME = os.getenv("CATALOGO_FILENAME", "catalogo_freud.jsonl")
RAG_FILENAME      = os.getenv("RAG_FILENAME",      "freud_pages.jsonl")

USE_TIMESTAMPED_OUTPUT = os.getenv("USE_TIMESTAMPED_OUTPUT", "false").lower() in ("1","true","yes")
TIMESTAMP_FMT = os.getenv("TIMESTAMP_FMT", "%Y%m%d-%H%M%S")

def get_output_dir() -> Path:
    if USE_TIMESTAMPED_OUTPUT:
        stamp = datetime.now().strftime(TIMESTAMP_FMT)
        out = BUILD_DIR / stamp
    else:
        out = BUILD_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out

def get_catalogo_path() -> Path:
    return get_output_dir() / CATALOGO_FILENAME

def get_rag_pages_path() -> Path:
    return get_output_dir() / RAG_FILENAME

# Ollama
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

# Rete
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_SEC  = float(os.getenv("BACKOFF_SEC", "1.5"))


# ================================================================
#  ChromaDB — normalizzazione path rispetto alla ROOT
#  (se nel .env scrivi "./build/chroma_freud", diventa freud-kb/build/chroma_freud)
# ================================================================
CHROMA_PATH      = PROJECT_DIR / os.getenv("CHROMA_PATH", "build/chroma_freud")
CHROMA_DOCS_PATH = PROJECT_DIR / os.getenv("CHROMA_DOCS_PATH", "build/chroma_freud_docs")


# Crea le cartelle se non esistono
BUILD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)