# scripts/03b_index_documents.py
# Indicizzazione RAG — Documenti (collection separata)
# - Legge l'ultimo build/**/documents_index.jsonl prodotto da 02_download_documents.py
# - Apre SOLO i documenti scaricati (downloaded=true)
# - Estrae testo da PDF (pdfplumber), TXT (lettura diretta), DOCX (opzionale python-docx)
# - Chunkizza e genera embeddings via Ollama (batch)
# - Indicizza nella collection separata: freud_docs (in ./build/chroma_freud_docs)
#
# Esecuzione:
#   python scripts/03b_index_documents.py
#
# Output:
#   build/chroma_freud_docs/  (collection: freud_docs)

import os
import re
import io
import json
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm
import chromadb
from chromadb.config import Settings





# PDF
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# DOCX (opzionale)
try:
    import docx  # python-docx
except Exception:
    docx = None

from common_config import (
    BUILD_DIR, DATA_DIR,
    OLLAMA_URL, EMBED_MODEL, EMBED_BATCH,CHROMA_DOCS_PATH
)

# ------------------------------------------------------------------------------
# Config (.env)
# ------------------------------------------------------------------------------
#CHROMA_DOCS_PATH = Path(os.getenv("CHROMA_DOCS_PATH", BUILD_DIR / "chroma_freud_docs"))
COLLECTION_DOCS  = os.getenv("COLLECTION_DOCS", "freud_docs")

TARGET_CHARS   = int(os.getenv("CHUNK_TARGET_CHARS", "1000"))
MIN_CHARS      = int(os.getenv("CHUNK_MIN_CHARS", "200"))
OVERLAP_CHARS  = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))

DOCS_TEXT_EXTS = set(e.strip().lower() for e in os.getenv("DOCS_TEXT_EXTS", "pdf,txt,docx").split(",") if e.strip())
RECREATE       = os.getenv("DOCS_RECREATE_COLLECTION", "true").lower() in ("1","true","yes")

def _sanitize_for_embed(s: str) -> str:
    """
    Sanitizza leggermente il testo: comprime whitespace, limita la lunghezza
    e rimuove caratteri di controllo che in alcuni build di Ollama causano embedding vuoti.
    """
    if not s:
        return ""
    # comprime whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # rimuove caratteri di controllo non stampabili
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", s)
    # hard cap ragionevole per modelli embedding locali (evita payload enormi)
    MAX_CHARS_EMBED = 4000
    if len(s) > MAX_CHARS_EMBED:
        s = s[:MAX_CHARS_EMBED]
    return s

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def chunk_text(text: str, target: int = TARGET_CHARS, min_chars: int = MIN_CHARS, overlap: int = OVERLAP_CHARS):
    """Chunking a finestra; tenta di chiudere a confine frase per evitare tagli duri."""
    txt = (text or "").strip()
    n = len(txt)

    if overlap >= target:
        raise ValueError("overlap must be smaller than target")

    if n < min_chars:
        return [txt]
    if n <= target:
        return [txt]

    out = []
    start = 0

    while start < n:
        stop = min(start + target, n)
        ext_slice = txt[stop: stop + 100]
        m = re.search(r"([\.!?])\s+[A-ZÀ-ÖØ-Ý]", ext_slice)
        if m:
            stop = stop + m.start() + 1

        chunk = txt[start:stop].strip()
        if len(chunk) >= min_chars:
            out.append(chunk)

        if stop >= n:
            break

        start = max(stop - overlap, 0)

    return out

def build_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "FreudKB/DocsIndexer"})
    return s

def try_parse_embedding(js):
    # Restituisce un embedding da varie forme possibili
    if isinstance(js, dict):
        if isinstance(js.get("embedding"), list):
            return js["embedding"]
        data = js.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            emb = data[0].get("embedding")
            if isinstance(emb, list):
                return emb
    return None

def embed_batch(session, texts):
    """
    Batch embedding robusto alle varianti di payload e alle risposte di Ollama.
    Gestisce:
      - {"data":[{"embedding":[...]} , ...]}
      - {"embeddings":[ [...], [...], ... ]}
      - {"embedding":[...]} (solo singolo)
    Se il batch fallisce (embedding vuoto), degrada a mini-batch/singolo.
    """
    # 1) Sanitizza i testi prima di inviare
    texts = [_sanitize_for_embed(t or "") for t in texts]
    texts = [t for t in texts if t]  # filtra eventuali vuoti
    if not texts:
        return []

    def _parse_embeddings(js, expected_count):
        """
        Ritorna una lista di embeddings o None se il formato non combacia.
        """
        if not isinstance(js, dict):
            return None

        # Forma nuova: {"embeddings":[ [...], [...], ... ]}
        if "embeddings" in js and isinstance(js["embeddings"], list):
            embs = [e for e in js["embeddings"] if isinstance(e, list)]
            if len(embs) == expected_count:
                return embs

        # Forma storica: {"data":[ {"embedding":[...]}, ... ]}
        if "data" in js and isinstance(js["data"], list):
            embs = []
            for item in js["data"]:
                if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                    embs.append(item["embedding"])
            if len(embs) == expected_count:
                return embs

        # Forma singola: {"embedding":[...]} (solo se expected_count==1)
        if expected_count == 1 and isinstance(js.get("embedding"), list) and js["embedding"]:
            return [js["embedding"]]

        return None

    # 2) Primo tentativo: invio in batch (input/lista)
    batch_payloads = [
        {"model": EMBED_MODEL, "input": texts},   # forma raccomandata
        {"model": EMBED_MODEL, "prompt": texts},  # fallback legacy
    ]
    last = None
    for p in batch_payloads:
        try:
            r = session.post(OLLAMA_URL, json=p, timeout=180)
            r.raise_for_status()
            js = r.json()
            embs = _parse_embeddings(js, expected_count=len(texts))
            if embs:
                return embs
            last = js
        except Exception:
            pass  # proveremo mini-batch

    # 3) Degrado a mini-batch (per evitare casi in cui il batch restituisce embedding vuoti)
    MINI = 8  # mini-batch di sicurezza
    out_embs = []
    for i in range(0, len(texts), MINI):
        mini = texts[i:i+MINI]
        ok = False
        for p in batch_payloads:
            try:
                # aggiorna il payload con la mini-list
                p2 = dict(p)
                if "input" in p2:
                    p2["input"] = mini
                    p2.pop("prompt", None)
                elif "prompt" in p2:
                    p2["prompt"] = mini
                    p2.pop("input", None)

                r = session.post(OLLAMA_URL, json=p2, timeout=180)
                r.raise_for_status()
                js = r.json()
                embs = _parse_embeddings(js, expected_count=len(mini))
                if embs:
                    out_embs.extend(embs)
                    ok = True
                    break
                last = js
            except Exception:
                pass
        if not ok:
            # 4) Degrado ulteriore: invio 1-per-1 (massima compatibilità)
            for t in mini:
                single_payloads = [
                    {"model": EMBED_MODEL, "input": [t]},
                    {"model": EMBED_MODEL, "prompt": [t]},
                    {"model": EMBED_MODEL, "input": t},
                    {"model": EMBED_MODEL, "prompt": t},
                ]
                got = False
                for sp in single_payloads:
                    try:
                        r = session.post(OLLAMA_URL, json=sp, timeout=180)
                        r.raise_for_status()
                        js = r.json()
                        embs = _parse_embeddings(js, expected_count=1)
                        if embs and isinstance(embs[0], list):
                            out_embs.append(embs[0])
                            got = True
                            break
                        last = js
                    except Exception:
                        pass
                if not got:
                    raise RuntimeError(f"Embedding non disponibile anche in singolo. Ultima risposta: {last}")

    if len(out_embs) != len(texts):
        raise RuntimeError(f"Embedding incompleti: attesi {len(texts)}, ottenuti {len(out_embs)}. Ultima risposta: {last}")

    return out_embs

# def recreate_collection(path: Path, name: str):
#     client = chromadb.PersistentClient(path=str(path), settings=Settings(allow_reset=True))
#     if RECREATE:
#         try:
#             client.delete_collection(name)
#         except Exception:
#             pass
#     # se esiste e non l'abbiamo cancellata, get_collection; altrimenti create
#     try:
#         return client.get_collection(name)
#     except Exception:
#         return client.create_collection(name)

def recreate_collection(path: Path, name: str):
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(path), settings=Settings(allow_reset=True))
    if RECREATE:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    # Se esiste e non l'abbiamo cancellata, get_collection; altrimenti create (cosine)
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

# ------------------------------------------------------------------------------
# Reading latest documents_index.jsonl
# ------------------------------------------------------------------------------
def latest_documents_index() -> Path | None:
    candidates = sorted(BUILD_DIR.glob("**/documents_index.jsonl"))
    return candidates[-1] if candidates else None

# ------------------------------------------------------------------------------
# Text extraction
# ------------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    if not pdfplumber:
        return ""
    try:
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                if t:
                    pages.append(t)
        text = re.sub(r"\s+", " ", " ".join(pages)).strip()
        return text
    except Exception:
        return ""

def extract_text_from_txt(txt_path: Path) -> str:
    try:
        return re.sub(r"\s+", " ", txt_path.read_text(encoding="utf-8", errors="ignore")).strip()
    except Exception:
        return ""

def extract_text_from_docx(docx_path: Path) -> str:
    if not docx:
        return ""
    try:
        d = docx.Document(str(docx_path))
        parts = []
        for p in d.paragraphs:
            t = (p.text or "").strip()
            if t:
                parts.append(t)
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        return text
    except Exception:
        return ""

# ------------------------------------------------------------------------------
# Load docs and build "pages-like" entries
# ------------------------------------------------------------------------------
def load_docs_as_pages() -> list[dict]:
    idx = latest_documents_index()
    if not idx or not idx.exists():
        print(">> Nessun documents_index.jsonl trovato: esci.")
        return []

    print(f">> Index documenti: {idx}")
    pages = []
    with idx.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if not obj.get("downloaded"):
                continue

            url   = obj.get("url") or ""
            title = (obj.get("filename_original") or Path(url).name or "Documento").strip()
            rel   = obj.get("saved_path") or ""
            ext   = (obj.get("file_ext") or "").lower()

            if not rel:
                continue
            if ext not in DOCS_TEXT_EXTS:
                continue

            file_path = (DATA_DIR / rel).resolve()
            if not file_path.exists():
                continue

            # Estrazione testo in base al tipo
            text = ""
            if ext == "pdf":
                text = extract_text_from_pdf(file_path)
            elif ext == "txt":
                text = extract_text_from_txt(file_path)
            elif ext == "docx":
                text = extract_text_from_docx(file_path)
            else:
                # altri tipi non trattati per ora
                continue

            if not text or len(text) < MIN_CHARS:
                continue

            pid = hashlib.md5((url or str(file_path)).encode("utf-8")).hexdigest()
            pages.append({
                "id": pid,
                "url": url,
                "title": title,
                "content": text
            })

    print(f">> Documenti testuali caricati: {len(pages)} (estensioni: {sorted(DOCS_TEXT_EXTS)})")
    return pages

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    print(">> Avvio indicizzazione DOCS (collection separata)")
    print(f">> Collection: {COLLECTION_DOCS}  |  Path: {CHROMA_DOCS_PATH}")
    print(f">> Chunking: target={TARGET_CHARS}, min={MIN_CHARS}, overlap={OVERLAP_CHARS}")
    print(f">> Embedding model: {EMBED_MODEL}  |  Batch size: {EMBED_BATCH}")
    print(f">> Estensioni testuali: {sorted(DOCS_TEXT_EXTS)}  |  Recreate={RECREATE}")

    pages = load_docs_as_pages()
    if not pages:
        print(">> Nessun documento testuale da indicizzare. Fine.")
        return

    # Chunking
    chunks = []
    for p in pages:
        cs = chunk_text(p["content"])
        for i, ch in enumerate(cs):
            cid = f"{p['id']}::c{i:03d}"
            chunks.append({
                "id": cid,
                "text": ch,
                "url": p["url"],
                "title": p["title"],
                "page_id": p["id"]
            })
    print(f">> Chunk generati (DOCS): {len(chunks)}")
    if not chunks:
        print(">> Nessun chunk generato dai documenti. Fine.")
        return

    # Collection
    print(">> Creo/ricreo la collection DOCS …")
    col = recreate_collection(CHROMA_DOCS_PATH, COLLECTION_DOCS)

    # Embedding + add
    sess = build_session()
    total = 0
    for i in tqdm(range(0, len(chunks), EMBED_BATCH), desc="Indicizzazione DOCS"):
        batch = chunks[i:i+EMBED_BATCH]
        texts = [c["text"] for c in batch]

        try:
            embs = embed_batch(sess, texts)
        except Exception as e:
            raise SystemExit(f"[ERRORE] Generazione embeddings DOCS fallita: {e}")

        ids = [c["id"] for c in batch]
        metadatas = [{"url": c["url"], "title": c["title"], "page_id": c["page_id"]} for c in batch]

        col.add(
            ids=ids,
            embeddings=embs,
            documents=texts,
            metadatas=metadatas
        )
        total += len(batch)

    print(f">> COMPLETATO: {total} chunk indicizzati (DOCS).")
    print(f">> DB DOCS pronto in: {CHROMA_DOCS_PATH}  |  Collection: {COLLECTION_DOCS}")

if __name__ == "__main__":
    main()