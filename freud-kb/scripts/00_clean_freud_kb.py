# scripts/00_clean_freud_kb.py
# Pulisce in sicurezza tutti i file rigenerabili della pipeline Freud-KB
# Funziona correttamente anche su Windows + VS Code (rimozione forzata permessi)

import os
import stat
import shutil
from pathlib import Path

# ROOT = freud-kb/
ROOT = Path(__file__).resolve().parent.parent

# BUILD dirs
BUILD = ROOT / "build"
PAGES = BUILD / "pages"
CHROMA = BUILD / "chroma_freud"

# DATA dirs (NUOVO)
DOCUMENTS = ROOT / "data" / "documents"

def force_delete(action, name, exc):
    """
    Callback per rimuovere permessi (Windows) e forzare delete.
    """
    try:
        os.chmod(name, stat.S_IWRITE)
    except Exception:
        pass
    try:
        action(name)
    except Exception:
        pass

def safe_rmdir(p: Path):
    if p.exists():
        shutil.rmtree(p, onerror=force_delete)

def safe_rm(p: Path):
    if p.exists():
        try:
            os.chmod(p, stat.S_IWRITE)
        except Exception:
            pass
        try:
            p.unlink()
        except Exception:
            pass

def main():
    print("=== PULIZIA FREUD-KB (Windows-friendly) ===\n")

    # 1 — Cancella pages/
    print("[1] Pulizia cartella build/pages/")
    safe_rmdir(PAGES)
    PAGES.mkdir(parents=True, exist_ok=True)


    # 2 — Cancella chroma_freud/
    print("[2] Pulizia ChromaDB HTML (build/chroma_freud/)")
    safe_rmdir(CHROMA)
    CHROMA.mkdir(parents=True, exist_ok=True)

    # 2b — Cancella chroma_freud_docs/
    CHROMA_DOCS = BUILD / "chroma_freud_docs"
    print("[2b] Pulizia ChromaDB DOCUMENTI (build/chroma_freud_docs/)")
    safe_rmdir(CHROMA_DOCS)
    CHROMA_DOCS.mkdir(parents=True, exist_ok=True)


    # 3 — Rimuovi vecchi JSON della vecchia pipeline
    print("[3] Rimozione file JSON vecchi (kb_pages, kb_chunks)")
    safe_rm(BUILD / "kb_pages.jsonl")
    safe_rm(BUILD / "kb_chunks.jsonl")

    # 4 — Rimuovi cartelle timestamp
    print("[4] Pulizia cartelle timestamp in build/")
    for sub in BUILD.iterdir():
        if sub.is_dir() and sub.name[0].isdigit():
            print(f"   - Rimuovo {sub}")
            safe_rmdir(sub)

    # 5 — NUOVO: pulizia cartella documenti scaricati
    print("[5] Pulizia cartella data/documents/")
    safe_rmdir(DOCUMENTS)
    DOCUMENTS.mkdir(parents=True, exist_ok=True)

    print("\n✔ COMPLETATO — ambiente pulito e pronto.")

if __name__ == "__main__":
    main()