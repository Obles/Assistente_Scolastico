import os
import sys
import re
import argparse
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

CONTACT_KEYWORDS = {
    "contatti",
    "contatto",
    "segreteria",
    "telefono",
    "telefoni",
    "email",
    "mail",
    "fax",
    "orari",
    "orario",
    "didattica",
    "segreterie",
}

CONTACT_SECTIONS = {
    "Contatti Segreteria",
    "Organigramma",
    "Contatti",
}


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
    retrieval_threshold = float(os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.60"))

    return {
        "build_dir": str((project_root / build_dir).resolve()),
        "chroma_path": str((project_root / chroma_path).resolve()),
        "chroma_docs_path": str((project_root / chroma_docs_path).resolve()),
        "collection": collection,
        "collection_docs": collection_docs,
        "top_k_html": top_k_html,
        "top_k_docs": top_k_docs,
        "top_k_global": top_k_global,
        "retrieval_threshold": retrieval_threshold,
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
                "score": dist,
                "document": doc,
                "meta": meta or {},
            })
        return out
    except Exception as e:
        err(f"Errore query collection {source_label}: {e}")
        return []


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def detect_contact_query(query: str) -> bool:
    q = normalize_spaces(query).lower()
    return any(k in q for k in CONTACT_KEYWORDS)


def section_of(row: dict) -> str:
    meta = row.get("meta") or {}
    return (meta.get("section") or "").strip()


def title_of(row: dict) -> str:
    meta = row.get("meta") or {}
    return (meta.get("title") or "").strip()


def url_of(row: dict) -> str:
    meta = row.get("meta") or {}
    return (meta.get("url") or "").strip()


def filter_by_section(results: list, section_value: str) -> list:
    if not section_value:
        return results

    wanted = section_value.strip().lower()
    out = []
    for row in results:
        sec = section_of(row).lower()
        if sec == wanted:
            out.append(row)
    return out


def apply_threshold(results: list, threshold: float) -> list:
    return [r for r in results if float(r.get("distance", 999.0)) <= threshold]


def compute_score(row: dict, prefer_html: bool, contact_mode: bool) -> float:
    """
    score più basso = migliore.
    Base = distance; poi applichiamo piccoli boost/penalty.
    """
    score = float(row["distance"])
    source = row["source"]
    sec = section_of(row)

    # preferenza generale per html
    if prefer_html and source == "html":
        score -= 0.020
    elif prefer_html and source == "docs":
        score += 0.020

    # modalità contatti: HTML molto preferito, DOCS penalizzati
    if contact_mode:
        if source == "html":
            score -= 0.040
            if sec in CONTACT_SECTIONS:
                score -= 0.080
        elif source == "docs":
            score += 0.080

        # piccolo boost se nel documento compaiono parole contatto
        text = normalize_spaces(row.get("document", "")).lower()
        if any(k in text for k in ("telefono", "email", "mail", "fax", "segreteria", "contatti")):
            score -= 0.015

    return score


def rerank_results(html_results: list, docs_results: list, prefer_html: bool, contact_mode: bool, top_k_global: int):
    fused = html_results + docs_results

    for row in fused:
        row["score"] = compute_score(row, prefer_html=prefer_html, contact_mode=contact_mode)

    fused = sorted(fused, key=lambda x: (x["score"], x["distance"]))
    return fused[: max(1, top_k_global)]


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
        score = row.get("score", dist)
        sec = meta.get("section", "")
        snippet = normalize_spaces(row["document"])[:220]

        extra = f" section={sec}" if sec else ""
        print(f"{i}. [{row['source']}] dist={dist:.3f} score={score:.3f}{extra} {title_} — {url}")
        print("   ", snippet, "…")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Test retrieval fuso HTML + DOCS per Freud-KB"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Query testuale libera",
    )
    parser.add_argument(
        "--section",
        dest="section",
        default="",
        help='Filtra i risultati HTML per il metadato section (es. "Contatti Segreteria")',
    )
    parser.add_argument(
        "--prefer-html",
        dest="prefer_html",
        action="store_true",
        help="Favorisce i risultati HTML nel ranking finale",
    )
    parser.add_argument(
        "--contact-mode",
        dest="contact_mode",
        action="store_true",
        help="Modalità ottimizzata per query contatti/segreteria",
    )
    return parser


def main():
    print(BANNER)

    parser = build_arg_parser()
    args = parser.parse_args()

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

    query = " ".join(args.query).strip() or DEFAULT_QUERY
    auto_contact_mode = detect_contact_query(query)
    contact_mode = args.contact_mode or auto_contact_mode
    prefer_html = args.prefer_html or contact_mode

    print("Query:", query)
    if args.section:
        print("Filtro section HTML:", args.section)
    print("Prefer HTML:", prefer_html)
    print("Contact mode:", contact_mode)
    print("Threshold:", paths["retrieval_threshold"])

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

    # allargato un po' per fare tuning migliore prima del reranking
    html_results = query_collection(col_html, emb, max(paths["top_k_html"], 5), "html")
    docs_results = query_collection(col_docs, emb, max(paths["top_k_docs"], 3), "docs")

    # soglia base
    html_results = apply_threshold(html_results, paths["retrieval_threshold"])
    docs_results = apply_threshold(docs_results, paths["retrieval_threshold"])

    # filtro sezione solo sugli HTML
    if args.section:
        html_results = filter_by_section(html_results, args.section)

    print_results("RISULTATI HTML", html_results)
    print_results("RISULTATI DOCS", docs_results)

    fused = rerank_results(
        html_results=html_results,
        docs_results=docs_results,
        prefer_html=prefer_html,
        contact_mode=contact_mode,
        top_k_global=max(paths["top_k_global"], 5),
    )

    print_results("RISULTATI FUSI", fused)

    if not fused:
        warn("Nessun risultato nel retrieval fuso")
        sys.exit(30)

    best = fused[0]["distance"]
    best_score = fused[0]["score"]
    best_source = fused[0]["source"]
    best_section = section_of(fused[0])

    ok(f"Best fused match = dist {best:.3f} / score {best_score:.3f} da sorgente '{best_source}'")
    if best_section:
        ok(f"Best fused section = {best_section}")
    ok("TEST COMPLETATO: retrieval fuso HTML + DOCS operativo")


if __name__ == "__main__":
    main()