# test_rag.py
# Verifica end-to-end per Freud-KB (indice Chroma + Ollama embeddings)
# Uso:  python test_rag.py [query opzionale]
import os
import sys
import json
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
import chromadb


BANNER = """
============================================================
  FREUD-KB — TEST RAG (Chroma + Ollama)
============================================================
""".strip()

DEFAULT_QUERY = "contatti segreteria"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def info(msg: str):
    print(f"[INFO] {msg}")


def ok(msg: str):
    print(f"[OK]   {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def err(msg: str):
    print(f"[ERR]  {msg}")


def read_env():
    """
    Carica .env partendo dalla root del progetto (file fuori da /scripts).
    - Se esegui da freud-kb/scripts/test_rag.py, carica ../.env
    """
    here = Path(__file__).resolve().parent       # .../scripts
    project_root = here.parent                   # .../
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        warn(".env non trovato nella root progetto; uso variabili d'ambiente correnti")
    return project_root


def get_paths_from_env(project_root: Path):
    build_dir = os.getenv('BUILD_DIR', 'build')
    chroma_path = os.getenv('CHROMA_PATH', 'build/chroma_freud')
    chroma_docs_path = os.getenv('CHROMA_DOCS_PATH', 'build/chroma_freud_docs')
    collection = os.getenv('COLLECTION', 'freud_kb')
    collection_docs = os.getenv('COLLECTION_DOCS', 'freud_docs')
    return {
        'build_dir': str((project_root / build_dir).resolve()),
        'chroma_path': str((project_root / chroma_path).resolve()),
        'chroma_docs_path': str((project_root / chroma_docs_path).resolve()),
        'collection': collection,
        'collection_docs': collection_docs,
        'project_root': str(project_root.resolve()),
    }


def healthcheck_ollama_tags(timeout=8) -> bool:
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=timeout)
        return r.ok
    except Exception:
        return False


def embed_ollama(text: str) -> list:
    """
    Richiede l'embedding a Ollama usando ESCLUSIVAMENTE l'endpoint /api/embed.
    - Usa OLLAMA_URL e EMBED_MODEL dal .env (valori di default inclusi).
    - Accetta i tre formati-ritorno comuni di Ollama:
        1) {"embeddings": [[...]]}
        2) {"data": [{"embedding": [...]}]}
        3) {"embedding": [...]}
    - Non applica fallback ad altri endpoint.
    Ritorna: lista di float (embedding) oppure solleva RuntimeError con dettaglio.
    """
    url = (os.getenv('OLLAMA_URL') or 'http://localhost:11434/api/embed').strip()
    model = os.getenv('EMBED_MODEL', 'nomic-embed-text')

    # Payload "moderno": input lista (consigliato da Ollama)
    payloads = [
        {"model": model, "input": [text]},
        # Compatibilità legacy (alcune build accettano 'prompt' anziché 'input')
        {"model": model, "prompt": [text]},
    ]

    last = None
    for payload in payloads:
        try:
            r = requests.post(url, json=payload, timeout=60)
            if not r.ok:
                last = {"status": r.status_code, "text": (r.text or "")[:400]}
                continue
            js = r.json()
            last = js

            # 1) {"embeddings": [[...]]}
            embs = js.get("embeddings")
            if isinstance(embs, list) and embs and isinstance(embs[0], list):
                return embs[0]

            # 2) {"data": [{"embedding": [...]}]}
            data = js.get("data")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if isinstance(data[0].get("embedding"), list):
                    return data[0]["embedding"]

            # 3) {"embedding": [...]}
            if isinstance(js.get("embedding"), list):
                return js["embedding"]

        except requests.Timeout:
            last = {"error": "timeout", "endpoint": url}
            continue
        except Exception as e:
            last = {"error": str(e), "endpoint": url}
            continue

    # Se arrivo qui, nessun payload ha funzionato
    raise RuntimeError(f"Embedding non disponibile da {url}. Ultima risposta: {last}")


def test_chroma(paths: dict, query: str):
    info("Connessione a Chroma (HTML)…")
    client = chromadb.PersistentClient(path=paths['chroma_path'])
    try:
        col = client.get_collection(paths['collection'])
    except Exception as e:
        err(f"Collection '{paths['collection']}' non trovata: {e}")
        return 2

    meta = col.metadata or {}
    print("\n== METADATA COLLECTION ==\n", meta)
    if meta.get('hnsw:space') == 'cosine':
        ok("Metrica: COSINE (corretta)")
    else:
        warn("Metrica NON cosine — ricrea l'indice con hnsw:space=cosine e cancella la cartella fisicamente")

    info("Generazione embedding query via Ollama…")
    emb = embed_ollama(query)
    print(f"Lunghezza embedding: {len(emb)}")
    if len(emb) != 768:
        warn("Embedding di lunghezza diversa da 768: verifica modello 'nomic-embed-text' e endpoint /api/embed")
        return 3

    info("Query su Chroma (top‑5 con distances)…")
    res = col.query(query_embeddings=[emb], n_results=5, include=["documents", "metadatas", "distances"])
    dists = res.get('distances') or [[]]
    docs = res.get('documents') or [[]]
    metas = res.get('metadatas') or [[]]
    if not docs[0]:
        warn("Nessun risultato di retrieval")
        return 4

    print("\n== RISULTATI (top‑5) ==")
    for i, (dist, doc, meta) in enumerate(zip(dists[0], docs[0], metas[0]), start=1):
        url = (meta or {}).get('url', '')
        title = (meta or {}).get('title', '')
        print(f"{i}. dist={dist:.3f} {title} — {url}")
        snippet = re.sub(r"\s+", " ", (doc or ""))[:200]
        print("   ", snippet, "…")

    best = dists[0][0]
    ok(f"Best‑match distance = {best:.3f}  (più basso = migliore)")
    return 0


def main():
    print(BANNER)
    project_root = read_env()
    paths = get_paths_from_env(project_root)
    print("Root:", paths['project_root'])
    print("Chroma HTML:", paths['chroma_path'])

    # 0) Healthcheck Ollama
    info("Verifica Ollama /api/tags…")
    if not healthcheck_ollama_tags():
        warn("Ollama non risponde su http://localhost:11434 — avvia Ollama Desktop/servizio e riprova")
    else:
        ok("Ollama raggiungibile")

    # 1) Test Chroma + query
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    code = test_chroma(paths, query)

    print("\n============================================================")
    if code == 0:
        ok("TEST COMPLETATO: Retrieval operativo e metrica cosine corretta")
    else:
        warn(f"TEST CONCLUSO con codice {code}. Vedi messaggi sopra per la diagnosi.")


if __name__ == "__main__":
    main()