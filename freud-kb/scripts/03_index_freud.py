# scripts/03_index_freud.py
# Indicizzazione RAG per Freud-KB:
# - Legge freud_pages.jsonl (contenuti estratti dallo scraper A+)
# - Deduplica, chunkizza (800–1200, overlap 120 per default)
# - Genera embeddings via Ollama in batch (robusto a formati/mini/singolo)
# - Ricrea ChromaDB (collection freud_kb) con metrica COSINE
# - *** NOVITÀ ***: Propaga 'section' su ogni chunk e lo salva nei metadati

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Tuple

import requests
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

from common_config import (
    BUILD_DIR, OLLAMA_URL, EMBED_MODEL, EMBED_BATCH, CHROMA_PATH
)

COLLECTION  = os.getenv("COLLECTION", "freud_kb")

# ---------------------------------------------------------------------
# Config path/collection (override da .env se servono)
# ---------------------------------------------------------------------
RAG_INPUT = os.getenv("RAG_INPUT_FILE", "").strip()
if RAG_INPUT:
    RAG_FILE = Path(RAG_INPUT)
else:
    candidates = sorted(BUILD_DIR.glob("**/freud_pages.jsonl"))
    RAG_FILE = candidates[-1] if candidates else (BUILD_DIR / "freud_pages.jsonl")

if RAG_FILE.is_dir():
    raise SystemExit(
        f"[ERRORE] RAG_INPUT_FILE punta a una CARTELLA: {RAG_FILE}\n"
        "Devi indicare un file .jsonl valido (es: build/<timestamp>/freud_pages.jsonl)."
    )
if not RAG_FILE.exists():
    raise SystemExit(
        f"[ERRORE] File RAG non trovato: {RAG_FILE}\n"
        "Assicurati di aver eseguito prima scripts/01_scrape_freud.py oppure imposta RAG_INPUT_FILE al file corretto."
    )

TARGET_CHARS   = int(os.getenv("CHUNK_TARGET_CHARS", "1000"))
MIN_CHARS      = int(os.getenv("CHUNK_MIN_CHARS", "200"))
OVERLAP_CHARS  = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))

# ---------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------
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

def dedup_texts(pages):
    seen = set()
    unique = []
    for p in pages:
        key = hashlib.md5((p["title"].lower() + "\n" + p["content"].lower()).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique

def build_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "FreudKB/Indexer"})
    return s

def _sanitize_for_embed(s: str) -> str:
    """Sanitizza leggermente il testo per embedding locali."""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", s)
    MAX_CHARS_EMBED = 4000
    if len(s) > MAX_CHARS_EMBED:
        s = s[:MAX_CHARS_EMBED]
    return s

def embed_batch(session, texts):
    """
    Batch embedding robusto a vari payload/risposte Ollama.
    Gestisce:
      - {"embeddings":[ [...], [...], ... ]}
      - {"data":[{"embedding":[...]} , ...]}
      - {"embedding":[...]} (singolo)
    Se batch fallisce → mini-batch → singolo.
    """
    texts = [_sanitize_for_embed(t or "") for t in texts]
    texts = [t for t in texts if t]
    if not texts:
        return []

    def _parse_embeddings(js, expected):
        if not isinstance(js, dict):
            return None
        if "embeddings" in js and isinstance(js["embeddings"], list):
            embs = [e for e in js["embeddings"] if isinstance(e, list)]
            return embs if len(embs) == expected else None
        if "data" in js and isinstance(js["data"], list):
            embs = []
            for item in js["data"]:
                if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                    embs.append(item["embedding"])
            return embs if len(embs) == expected else None
        if expected == 1 and isinstance(js.get("embedding"), list) and js["embedding"]:
            return [js["embedding"]]
        return None

    batch_payloads = [
        {"model": EMBED_MODEL, "input": texts},
        {"model": EMBED_MODEL, "prompt": texts},
    ]
    last = None
    for p in batch_payloads:
        try:
            r = session.post(OLLAMA_URL, json=p, timeout=180)
            r.raise_for_status()
            js = r.json()
            embs = _parse_embeddings(js, expected=len(texts))
            if embs:
                return embs
            last = js
        except Exception:
            pass

    MINI = 8
    out_embs = []
    for i in range(0, len(texts), MINI):
        mini = texts[i:i+MINI]
        ok = False
        for p in batch_payloads:
            try:
                p2 = dict(p)
                if "input" in p2:
                    p2["input"] = mini
                    p2.pop("prompt", None)
                if "prompt" in p2:
                    p2["prompt"] = mini
                    p2.pop("input", None)

                r = session.post(OLLAMA_URL, json=p2, timeout=180)
                r.raise_for_status()
                js = r.json()
                embs = _parse_embeddings(js, expected=len(mini))
                if embs:
                    out_embs.extend(embs)
                    ok = True
                    break
                last = js
            except Exception:
                pass
        if not ok:
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
                        embs = _parse_embeddings(js, expected=1)
                        if embs and isinstance(embs[0], list):
                            out_embs.append(embs[0])   # <<< QUI la fix
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

def recreate_collection(path: Path, name: str):
    client = chromadb.PersistentClient(path=str(path), settings=Settings(allow_reset=True))
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

# ---------------------------------------------------------------------
# Caricamento input
# ---------------------------------------------------------------------
def load_pages(rag_path: Path):
    if not rag_path.exists():
        raise SystemExit(f"[ERRORE] File RAG non trovato: {rag_path}\n"
                         f"Esegui prima scripts/01_scrape_freud.py")

    pages = []
    seen_ids = set()
    with rag_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid     = obj.get("id")
            url     = obj.get("url", "")
            title   = (obj.get("title") or "").strip()
            content = (obj.get("content") or "").strip()
            section = (obj.get("section") or "Generico").strip()   # <<<<<< 1) LEGGI SECTION
            if not pid or not content:
                continue
            if pid in seen_ids:
                continue
            pages.append({
                "id": pid,
                "url": url,
                "title": title,
                "content": content,
                "section": section                                  # <<<<<< conserva SECTION
            })
            seen_ids.add(pid)
    return pages

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print(">> Avvio indicizzazione RAG Freud")
    print(f">> Input: {RAG_FILE}")
    print(f">> Chroma path: {CHROMA_PATH}  |  Collection: {COLLECTION}")
    print(f">> Chunking: target={TARGET_CHARS}, min={MIN_CHARS}, overlap={OVERLAP_CHARS}")
    print(f">> Embedding model: {EMBED_MODEL}  |  Batch size: {EMBED_BATCH}")

    pages = load_pages(RAG_FILE)
    if not pages:
        raise SystemExit("[ERRORE] Nessuna pagina valida trovata in freud_pages.jsonl")

    pages = dedup_texts(pages)
    print(f">> Pagine uniche per indicizzazione: {len(pages)}")

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
                "page_id": p["id"],
                "section": p["section"]             # <<<<<< 2) PROPAGA SECTION
            })
    print(f">> Chunk generati: {len(chunks)}")
    if not chunks:
        raise SystemExit("[ERRORE] Nessun chunk generato (contenuti troppo corti?)")

    # Ricrea collection
    print(">> Creo/ricreo la collection Chroma …")
    col = recreate_collection(Path(CHROMA_PATH), COLLECTION)

    # Embedding + add
    sess = build_session()
    total = 0
    for i in tqdm(range(0, len(chunks), EMBED_BATCH), desc="Indicizzazione"):
        batch = chunks[i:i+EMBED_BATCH]
        texts = [c["text"] for c in batch]

        try:
            embs = embed_batch(sess, texts)
        except Exception as e:
            raise SystemExit(f"[ERRORE] Generazione embeddings fallita: {e}")

        ids = [c["id"] for c in batch]
        metadatas = [{
            "url": c["url"],
            "title": c["title"],
            "page_id": c["page_id"],
            "section": c.get("section", "Generico")  # <<<<<< 3) SALVA SECTION NEI METADATI
        } for c in batch]

        col.add(
            ids=ids,
            embeddings=embs,
            documents=texts,
            metadatas=metadatas
        )
        total += len(batch)

    print(f">> COMPLETATO: {total} chunk indicizzati.")
    print(f">> DB pronto in: {CHROMA_PATH}  |  Collection: {COLLECTION}")

if __name__ == "__main__":
    main()