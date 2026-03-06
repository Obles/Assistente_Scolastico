import os
import sys
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
import chromadb


BANNER = """
============================================================
  FREUD-KB — TEST RAG FUSO (HTML + DOCS)
============================================================
""".strip()

DEFAULT_QUERY = "contatti segreteria"


def info(msg: str):
    print(f"[INFO] {msg}")


def ok(msg: str):
    print(f"[OK]   {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def err(msg: str):
    print(f"[ERR]  {msg}")


def read_env():
    here = Path(__file__).resolve().parent
    project_root = here.parent if (here / ".env").exists() is False else here
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        warn(".env non trovato nella root progetto; uso variabili d'ambiente correnti")
    return project_root


def get_paths_from_env(project_root: Path):
    build_dir = os.getenv("BUILD_DIR", "build")
    chroma_path = os.getenv("CHROMA_PATH", "build/chroma_freud")
    chroma_docs_path = os.getenv("CHROMA_DOCS_PATH", "build/chroma_freud_docs")
    collection = os.getenv("COLLECTION", "freud_kb")
    collection_docs = os.getenv("COLLECTION_DOCS", "freud_docs")
    top_k_html = int(os.getenv("TOP_K_HTML", "3"))
    top_k_docs = int(os.getenv("TOP_K_DOCS", "3"))
    top_k_global = int(os.getenv("TOP_K_GLOBAL", "5"))

    return {
        "build_dir": str((project_root / build_dir).resolve()),
        "chroma_path": str((project_root / chroma_path).resolve()),
        "chroma_docs_path": str((project_root / chroma_docs_path).resolve()),
        "collection": collection,
        "collection_docs": collection_docs,
        "top_k_html": top_k_html,
        "top_k_docs": top_k_docs,
        "top_k_global": top_k_global,
        "project_root": str(project_root.resolve()),
    }


def healthcheck_ollama_tags(timeout=8) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=timeout)
        return r.ok
    except Exception:
        return False


def embed_ollama(text: str) -> list:
    """
    Richiede embedding a Ollama usando /api/embed.
    Accetta i formati:
      1) {"embeddings": [[...]]}
      2) {"data": [{"embedding": [...]}]}
      3) {"embedding": [...]}
    """
    url = (os.getenv("OLLAMA_URL") or "http://localhost:11434/api/embed").strip()
    model = os.getenv("EMBED_MODEL", "nomic-embed-text")

    payloads = [
        {"model": model, "input": [text]},
        {"model": model, "prompt": [text]},
        {"model": model, "input": text},
        {"model": model, "prompt": text},
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

            embs = js.get("embeddings")
            if isinstance(embs, list) and embs and isinstance(embs[0], list):
                return embs[0]

            data = js.get("data")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                emb = data[0].get("embedding")
                if isinstance(emb, list):
                    return emb

            emb = js.get("embedding")
            if isinstance(emb, list):
                return emb

        except requests.Timeout:
            last = {"error": "timeout", "endpoint": url}
        except Exception as e:
            last = {"error": str(e), "endpoint": url}

    raise RuntimeError(f"Embedding non disponibile da {url}. Ultima risposta: {last}")


def open_collection(chroma_path: str, collection_name: str, label: str):
    info(f"Connessione a Chroma ({label})…")
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        col = client.get_collection(collection_name)
        meta = col.metadata or {}
        print(f"\n== METADATA COLLECTION {label.upper()} ==\n", meta)
        if meta.get("hnsw:space") == "cosine":
            ok(f"Metrica {label}: COSINE")
        else:
            warn(f"Metrica {label} NON cosine")
        return col
    except Exception as e:
        err(f"Collection '{collection_name}' non trovata per {label}: {e}")
        return None


def query_collection(col, emb, n_results: int, source_label: str):
    if col is None:
        return []

    try:
        res = col.query(
            query_embeddings=[emb],
            n_results=max(1, n_results),
            include=["documents", "metadatas", "distances"],
        )

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out = []
        for i, doc in enumerate(docs):
            if not doc:
                continue
            meta = metas[i] if i < len(metas) else {}
            dist = float(dists[i]) if i < len(dists) else 999.0
            out.append({
                "source": source_label,
                "distance": dist,
                "document": doc,
                "meta": meta or {},
            })
        return out
    except Exception as e:
        err(f"Errore query collection {source_label}: {e}")
        return []


def print_results(title: str, results: list):
    print(f"\n== {title} ==")
    if not results:
        print("Nessun risultato")
        return

    for i, row in enumerate(results, start=1):
        meta = row["meta"]
        title_ = meta.get("title", "")
        url = meta.get("url", "")
        dist = row["distance"]
        snippet = re.sub(r"\s+", " ", row["document"])[:220]
        print(f"{i}. [{row['source']}] dist={dist:.3f} {title_} — {url}")
        print("   ", snippet, "…")


def main():
    print(BANNER)
    project_root = read_env()
    paths = get_paths_from_env(project_root)

    print("Root:", paths["project_root"])
    print("Chroma HTML:", paths["chroma_path"])
    print("Chroma DOCS:", paths["chroma_docs_path"])

    info("Verifica Ollama /api/tags…")
    if not healthcheck_ollama_tags():
        warn("Ollama non risponde su http://localhost:11434")
        sys.exit(10)
    ok("Ollama raggiungibile")

    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    print("Query:", query)

    info("Generazione embedding query via Ollama…")
    emb = embed_ollama(query)
    print(f"Lunghezza embedding: {len(emb)}")
    if len(emb) != 768:
        warn("Embedding di lunghezza diversa da 768: verifica modello nomic-embed-text e endpoint /api/embed")

    col_html = open_collection(paths["chroma_path"], paths["collection"], "html")
    col_docs = open_collection(paths["chroma_docs_path"], paths["collection_docs"], "docs")

    if col_html is None and col_docs is None:
        err("Nessuna collection disponibile")
        sys.exit(20)

    html_results = query_collection(col_html, emb, paths["top_k_html"], "html")
    docs_results = query_collection(col_docs, emb, paths["top_k_docs"], "docs")

    print_results("RISULTATI HTML", html_results)
    print_results("RISULTATI DOCS", docs_results)

    fused = sorted(
        html_results + docs_results,
        key=lambda x: x["distance"]
    )[: max(1, paths["top_k_global"])]

    print_results("RISULTATI FUSI", fused)

    if not fused:
        warn("Nessun risultato nel retrieval fuso")
        sys.exit(30)

    best = fused[0]["distance"]
    best_source = fused[0]["source"]
    ok(f"Best fused match = {best:.3f} da sorgente '{best_source}'")
    ok("TEST COMPLETATO: retrieval fuso HTML + DOCS operativo")


if __name__ == "__main__":
    main()