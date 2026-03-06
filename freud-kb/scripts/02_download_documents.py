# scripts/03_download_documents.py
# Document Scanner & Downloader — Produce:
#   (A) data/documents/*                        → file scaricati
#   (B) build/<timestamp>/documents_index.jsonl → indice metadati
#
# Caratteristiche:
# - Crawl HTML (BFS) limitato a dominio istitutofreud.it
# - Scoperta dinamica menu+sottomenu dalla homepage
# - Raccolta link a documenti per estensione nota
# - Filtro temporale (anno corrente / + anno precedente / tutti)
# - DRY-RUN con HEAD: analisi/anteprima (conteggi, anni, peso) senza scaricare
# - Download robusto (retry/backoff), hash SHA256, naming sicuro
# - DEBUG verboso e QUIET
#
# Esecuzione:
#   python scripts/03_download_documents.py

import os
import re
import io
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin, urlunparse, parse_qsl, urlencode, unquote
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# Config centralizzata dal progetto
from common_config import (
    DATA_DIR, DOCUMENTS_DIR, BUILD_DIR,
    HTTP_TIMEOUT, MAX_RETRIES, BACKOFF_SEC,
    get_output_dir
)

# ---------------------------------------------------------------------
# PARAMETRI DA .env
# ---------------------------------------------------------------------
DOMAIN = "istitutofreud.it"
HOME_URL = "https://www.istitutofreud.it/"

# Limiti e velocità (riutilizza variabili dello scraper)
MAX_DEPTH      = int(os.getenv("SCRAPER_MAX_DEPTH", "3"))
SLEEP_BETWEEN  = float(os.getenv("SCRAPER_SLEEP_SEC", "0.25"))
MAX_PAGES      = int(os.getenv("SCRAPER_MAX_PAGES", "1200"))

# Filtri URL
BLOCK_FRAGMENT    = True
ALLOW_QUERYSTRING_PAGES = os.getenv("SCRAPER_ALLOW_QUERYSTRING", "false").lower() in ("1","true","yes")
# Per i DOCUMENTI manteniamo SEMPRE la querystring (download reali)
KEEP_QUERYSTRING_FOR_DOCS = True

# Modalità
DEBUG_VERBOSE     = os.getenv("SCRAPER_DEBUG_VERBOSE", "false").lower() in ("1","true","yes")
DOCS_DRY_RUN      = os.getenv("DOCS_DRY_RUN", "false").lower() in ("1","true","yes")
SCRAPER_QUIET     = os.getenv("SCRAPER_QUIET", "false").lower() in ("1","true","yes")

# Estensioni consentite
DEFAULT_DOC_EXTS = "pdf,doc,docx,xls,xlsx,ppt,pptx,csv,zip,rtf,txt"
DOCS_EXTS = set(e.strip().lower() for e in os.getenv("DOCS_EXTENSIONS", DEFAULT_DOC_EXTS).split(",") if e.strip())

# Filtro temporale documenti
DOCS_YEAR_POLICY = os.getenv("DOCS_YEAR_POLICY", "current_and_previous").strip().lower()  # current | current_and_previous | all
DOCS_MIN_YEAR    = os.getenv("DOCS_MIN_YEAR", "").strip()  # opzionale alternativa
DOCS_ACCEPT_UNDATED = os.getenv("DOCS_ACCEPT_UNDATED", "true").lower() in ("1","true","yes")

UA = {"User-Agent": "Mozilla/5.0 (FreudKB-PCTO/document-downloader)"}

# Paths
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = get_output_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_INDEX_FILE = OUTPUT_DIR / "documents_index.jsonl"

# ---------------------------------------------------------------------
# HTTP Session
# ---------------------------------------------------------------------
def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        backoff_factor=BACKOFF_SEC,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    s.headers.update(UA)
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

SESSION = build_session()

# ---------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------
TRACKING_PREFIXES = ("utm_",)
TRACKING_PARAMS   = {"fbclid", "gclid", "mc_cid", "mc_eid", "pk_campaign", "pk_kwd"}

DISALLOWED_REGEXES = [
    r"/wp-json", r"/jsonapi", r"/search",
    r"/tag/", r"/categoria/", r"/category/", r"/author/",
    r"/feed/?$", r"/\?p=\d+",
    r"/page/\d+/?$",
]
DISALLOWED_RE = [re.compile(rx, re.I) for rx in DISALLOWED_REGEXES]

def _strip_tracking(q_items):
    out = []
    for k, v in q_items:
        kl = k.lower()
        if kl in TRACKING_PARAMS:
            continue
        if any(kl.startswith(pref) for pref in TRACKING_PREFIXES):
            continue
        out.append((k, v))
    return out

def _is_directory_like(path: str) -> bool:
    leaf = (path or "/").split("/")[-1]
    return "." not in leaf

def _host_in_domain(netloc: str) -> bool:
    host = (netloc or "").lower()
    return host.endswith(DOMAIN) or host == DOMAIN

def normalize_page_url(base: str, href: str) -> str | None:
    """Normalizza URL per pagine HTML: rimuove fragment, opzionalmente querystring."""
    if not href:
        return None
    try:
        abs_url = urljoin(base, href)
        u = urlparse(abs_url)
        if u.scheme not in ("http", "https"):
            return None
        if not _host_in_domain(u.netloc):
            return None

        query = ""
        if ALLOW_QUERYSTRING_PAGES:
            query = urlencode(_strip_tracking(parse_qsl(u.query, keep_blank_values=False)))

        fragment = "" if BLOCK_FRAGMENT else (u.fragment or "")
        path = re.sub(r"/{2,}", "/", u.path or "/")
        # trailing slash solo per directory-like
        if _is_directory_like(path) and not path.endswith("/"):
            path += "/"

        new = u._replace(query=query, fragment=fragment, path=path)
        return urlunparse(new)
    except Exception:
        return None

def normalize_doc_url(base: str, href: str) -> str | None:
    """Normalizza URL per documenti: rimuove fragment, MANTIENE querystring."""
    if not href:
        return None
    try:
        abs_url = urljoin(base, href)
        u = urlparse(abs_url)
        if u.scheme not in ("http", "https"):
            return None
        if not _host_in_domain(u.netloc):
            return None

        # Mantieni query (ripulita da tracking)
        query = urlencode(_strip_tracking(parse_qsl(u.query, keep_blank_values=False))) if KEEP_QUERYSTRING_FOR_DOCS else ""
        fragment = "" if BLOCK_FRAGMENT else (u.fragment or "")
        path = re.sub(r"/{2,}", "/", u.path or "/")

        new = u._replace(query=query, fragment=fragment, path=path)
        return urlunparse(new)
    except Exception:
        return None

def is_disallowed_by_pattern(path: str) -> bool:
    low = (path or "").lower()
    for rx in DISALLOWED_RE:
        if rx.search(low):
            return True
    return False

def extension_of_url(url: str) -> str:
    p = urlparse(url)
    name = Path(p.path).name.lower()
    if "." in name:
        return name.rsplit(".", 1)[-1]
    return ""

# ---------------------------------------------------------------------
# Date helpers (filtro anno)
# ---------------------------------------------------------------------
def _year_from_last_modified(headers) -> int | None:
    lm = headers.get("Last-Modified")
    if not lm:
        return None
    try:
        dt = parsedate_to_datetime(lm)
        if dt and not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.year if dt else None
    except Exception:
        return None

def _year_from_url(url: str) -> int | None:
    # cerca /2026/ nel path
    m = re.search(r"/(20\\d{2})/", url)
    if m:
        return int(m.group(1))
    # cerca 2026 nel filename
    fname = Path(urlparse(url).path).name.lower()
    m2 = re.search(r"(20\\d{2})", fname)
    if m2:
        return int(m2.group(1))
    return None

def _allowed_year(year: int | None) -> bool:
    if year is None:
        return DOCS_ACCEPT_UNDATED  # includi/escludi se non si riesce a stimare l'anno
    cur = datetime.now().year
    pol = DOCS_YEAR_POLICY
    if pol == "all":
        return True
    if pol == "current":
        return year == cur
    if pol == "current_and_previous":
        return year in {cur, cur - 1}
    # fallback su DOCS_MIN_YEAR se valorizzato
    if DOCS_MIN_YEAR:
        try:
            return year >= int(DOCS_MIN_YEAR)
        except Exception:
            pass
    # default conservativo
    return True

# ---------------------------------------------------------------------
# HEAD utils per analisi/filtri
# ---------------------------------------------------------------------
def head_for_meta(url: str) -> tuple[int | None, dict]:
    """Prova HEAD; se non supportato, prova GET (senza scaricare il body) per ottenere header."""
    try:
        r = SESSION.head(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
        if r.ok:
            return r.status_code, (r.headers or {})
        # se 405/404 o simili, fallback a GET (senza leggere il contenuto)
        r2 = SESSION.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True, stream=True)
        try:
            return r2.status_code, (r2.headers or {})
        finally:
            r2.close()
    except Exception:
        return None, {}

# ---------------------------------------------------------------------
# Menu + Sottomenu dinamici (homepage)
# ---------------------------------------------------------------------
def _collect_links_from_container(base_url: str, container) -> set[str]:
    urls = set()
    for a in container.find_all("a", href=True):
        n = normalize_page_url(base_url, a["href"])
        if n:
            urls.add(n)
    return urls

def discover_menu_and_submenus(home_url: str) -> set[str]:
    found = set()
    try:
        r = SESSION.get(home_url, timeout=HTTP_TIMEOUT)
        if not r.ok or not r.text:
            return {home_url}
        soup = BeautifulSoup(r.text, "lxml")

        candidates = []
        candidates += soup.find_all(["header", "nav", "footer"])
        candidates += soup.find_all(attrs={"role": re.compile(r"navigation", re.I)})
        candidates += soup.find_all("div", class_=re.compile(r"(menu|nav|dropdown|submenu|mega)", re.I))
        candidates += soup.find_all("ul",  class_=re.compile(r"(menu|nav|dropdown|submenu|mega)", re.I))
        candidates += soup.find_all("li",  class_=re.compile(r"(menu|nav|dropdown|submenu|menu-item|has-children)", re.I))

        for c in candidates:
            found |= _collect_links_from_container(home_url, c)

        found.add(home_url)

        if DEBUG_VERBOSE or DOCS_DRY_RUN:
            sample = sorted(list(found))[:20]
            tqdm.write(f">> (DOCS) Menu/Sottomenu trovati: {len(found)}")
            for u in sample:
                tqdm.write(f"   - {u}")
        return set(found)
    except Exception:
        return {home_url}

def prefixes_from_urls(urls: set[str]) -> set[str]:
    prefs = set()
    for u in urls:
        p = urlparse(u).path or "/"
        segs = [s for s in p.split("/") if s]
        if not segs:
            prefs.add("/")
        else:
            prefs.add("/" + segs[0])
    # rimuovi figli ridondanti
    cleaned = set()
    for p in sorted(prefs, key=len):
        if not any(p != q and p.startswith(q + "/") for q in prefs):
            cleaned.add(p)
    if not cleaned:
        cleaned.add("/")
    return cleaned

def is_allowed_by_prefix(path: str, allowed_prefixes: set[str]) -> bool:
    low = (path or "/").lower()
    if low == "/":
        return True
    for pref in allowed_prefixes:
        pl = pref.lower()
        if pl == "/":
            return True
        if low == pl or low.startswith(pl + "/") or low.startswith(pl + "?"):
            return True
    return False

# ---------------------------------------------------------------------
# Fetch utils
# ---------------------------------------------------------------------
def fetch_with_fallback(url: str) -> tuple[str, requests.Response | None]:
    try:
        r = SESSION.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
        if r.status_code == 404:
            u = urlparse(url)
            if u.path.endswith("/"):
                alt = urlunparse(u._replace(path=u.path[:-1] or "/"))
            else:
                alt = urlunparse(u._replace(path=(u.path + "/")))
            r2 = SESSION.get(alt, timeout=HTTP_TIMEOUT, allow_redirects=True)
            if r2.ok:
                return alt, r2
        return url, r
    except Exception:
        return url, None

# ---------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------
def write_jsonl(fp, obj: dict) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------
# Crawler per scoprire link a documenti
# ---------------------------------------------------------------------
def discover_document_links() -> list[tuple[str, str]]:
    """
    Ritorna lista di tuple (doc_url, referer_page).
    """
    visited_pages: set[str] = set()
    enqueued_pages: set[str] = set()
    doc_links: list[tuple[str, str]] = []

    # 1) Menu+sottomenu per prefissi dinamici
    menu_urls = discover_menu_and_submenus(HOME_URL)
    allowed_prefixes = prefixes_from_urls(menu_urls)
    tqdm.write(f">> (DOCS) Prefissi ammessi (dinamici): {sorted(allowed_prefixes)}")

    # 2) Seeds iniziali: home + menu
    seeds = [HOME_URL] + sorted(menu_urls)
    seeds = list(dict.fromkeys(seeds))

    # 3) BFS pagine HTML
    frontier: list[tuple[str,int]] = []
    for s in seeds:
        s_norm = normalize_page_url(s, s)
        if not s_norm:
            continue
        sp = urlparse(s_norm).path or "/"
        if is_allowed_by_prefix(sp, allowed_prefixes) and not is_disallowed_by_pattern(sp):
            if s_norm not in enqueued_pages:
                frontier.append((s_norm, 0))
                enqueued_pages.add(s_norm)

    processed = 0
    with tqdm(total=MAX_PAGES, unit="page", dynamic_ncols=True, desc="Scansione HTML (per link documenti)") as pbar:
        while frontier:
            page_url, depth = frontier.pop(0)
            if depth > MAX_DEPTH:
                continue
            if page_url in visited_pages:
                continue
            if processed >= MAX_PAGES:
                pbar.set_postfix_str(f"STOP MAX_PAGES={MAX_PAGES}")
                break

            fetch_url, resp = fetch_with_fallback(page_url)
            visited_pages.add(fetch_url)
            processed += 1
            pbar.update(1)

            if not resp or not resp.ok:
                time.sleep(SLEEP_BETWEEN)
                continue

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in ctype and "application/xhtml" not in ctype:
                time.sleep(SLEEP_BETWEEN)
                continue

            html = resp.text or ""
            soup = BeautifulSoup(html, "lxml")

            # 3.a) Estrai link a documenti
            for a in soup.find_all("a", href=True):
                href = a["href"]
                cand = normalize_doc_url(fetch_url, href)
                if not cand:
                    continue
                ext = extension_of_url(cand)
                if not ext:
                    continue
                if ext.lower() in DOCS_EXTS:
                    doc_links.append((cand, fetch_url))

            # 3.b) Espansione BFS per pagine HTML
            if depth < MAX_DEPTH:
                for a in soup.find_all("a", href=True):
                    n = normalize_page_url(fetch_url, a["href"])
                    if not n:
                        continue
                    path_n = urlparse(n).path or "/"
                    if not is_allowed_by_prefix(path_n, allowed_prefixes):
                        continue
                    if is_disallowed_by_pattern(path_n):
                        continue
                    if n not in visited_pages and n not in enqueued_pages:
                        frontier.append((n, depth + 1))
                        enqueued_pages.add(n)

            pbar.set_postfix_str(f"depth≤{depth}, found_docs={len(doc_links)}")
            time.sleep(SLEEP_BETWEEN)

    # dedup preservando ordine (per URL doc)
    seen = set()
    dedup: list[tuple[str,str]] = []
    for u, ref in doc_links:
        if u not in seen:
            dedup.append((u, ref))
            seen.add(u)

    if DEBUG_VERBOSE or DOCS_DRY_RUN:
        tqdm.write(f">> (DOCS) Link documenti scoperti (dedup): {len(dedup)}")
        for u, ref in dedup[:20]:
            tqdm.write(f"   - {u}  ←  {ref}")

    return dedup

# ---------------------------------------------------------------------
# DRY-RUN: analisi con HEAD (nessun download)
# ---------------------------------------------------------------------
def analyze_document_links(doc_links: list[tuple[str, str]]) -> None:
    stats_by_ext: dict[str,int] = {}
    stats_by_year: dict[str,int] = {}
    total_size = 0
    analyzed = 0
    would_download = 0

    with open(DOCS_INDEX_FILE, "w", encoding="utf-8") as index_out, \
         tqdm(total=len(doc_links), unit="file", dynamic_ncols=True, desc="Analisi documenti (HEAD)") as pbar:

        for doc_url, referer in doc_links:
            status = None
            headers = {}
            ext = extension_of_url(doc_url).lower()
            size = 0

            try:
                status, headers = head_for_meta(doc_url)
                if status:
                    size = int(headers.get("Content-Length", "0") or "0")
                    y_hdr = _year_from_last_modified(headers)
                    y_url = _year_from_url(doc_url)
                    year = y_hdr if y_hdr is not None else y_url
                    ok = _allowed_year(year)

                    # stats
                    analyzed += 1
                    stats_by_ext[ext] = stats_by_ext.get(ext, 0) + 1
                    key_year = str(year) if year is not None else "undated"
                    stats_by_year[key_year] = stats_by_year.get(key_year, 0) + 1
                    if ok:
                        would_download += 1
                        total_size += size

                    write_jsonl(index_out, {
                        "id": hashlib.md5(doc_url.encode()).hexdigest(),
                        "url": doc_url,
                        "referer": referer,
                        "file_ext": ext,
                        "content_type": (headers.get("Content-Type") or "").lower(),
                        "status_code": status,
                        "content_length": size,
                        "year": year,
                        "would_download": bool(ok),
                        "downloaded": False,
                        "saved_path": "",
                        "download_timestamp": int(time.time()),
                        "notes": "DRY-RUN: analisi HEAD"
                    })

            except Exception:
                write_jsonl(index_out, {
                    "id": hashlib.md5(doc_url.encode()).hexdigest(),
                    "url": doc_url,
                    "referer": referer,
                    "file_ext": ext,
                    "content_type": "",
                    "status_code": status,
                    "content_length": 0,
                    "year": None,
                    "would_download": False,
                    "downloaded": False,
                    "saved_path": "",
                    "download_timestamp": int(time.time()),
                    "notes": "DRY-RUN: errore HEAD/analisi"
                })
            finally:
                pbar.update(1)
                time.sleep(SLEEP_BETWEEN)

    # riepilogo
    tqdm.write(f">> Analizzati: {analyzed} | Scaricabili (policy): {would_download} | Peso stimato: {total_size/1024/1024:.2f} MB")
    if stats_by_ext:
        tqdm.write(">> Per estensione:")
        for k in sorted(stats_by_ext):
            tqdm.write(f"   - .{k}: {stats_by_ext[k]}")
    if stats_by_year:
        tqdm.write(">> Per anno:")
        for k in sorted(stats_by_year, reverse=True):
            tqdm.write(f"   - {k}: {stats_by_year[k]}")
    tqdm.write(f">> Indice (DRY-RUN): {DOCS_INDEX_FILE}")

# ---------------------------------------------------------------------
# Download documenti (con filtro anno)
# ---------------------------------------------------------------------
def safe_filename_from_url(url: str) -> str:
    """Crea un nome file leggibile e sicuro basato sull'URL (basename + estensione)."""
    p = urlparse(url)
    name = unquote(Path(p.path).name) or "document"
    # ✅ FIX: regex corretta (niente backslash doppi)
    name = re.sub(r"[^\w\.\-]+", "_", name)
    # fallback estensione se mancante
    if "." not in name:
        ext = extension_of_url(url)
        if ext:
            name = f"{name}.{ext}"
    return name

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download_documents(doc_links: list[tuple[str, str]]) -> None:
    # 🔒 Stop totale: se per errore questa funzione viene chiamata in dry-run, blocca tutto
    if DOCS_DRY_RUN:
        raise RuntimeError("DOCS_DRY_RUN=true: download disabilitato. Nessun file verrà creato.")

    saved = 0
    skipped = 0

    with open(DOCS_INDEX_FILE, "w", encoding="utf-8") as index_out, \
         tqdm(total=len(doc_links), unit="file", dynamic_ncols=True, desc="Download documenti") as pbar:

        for doc_url, referer in doc_links:
            status = None
            ctype = ""
            clen  = 0
            sha   = ""
            saved_path = ""
            filename_original = safe_filename_from_url(doc_url)
            ext = extension_of_url(doc_url).lower()
            year = None

            try:
                # 1) HEAD per valutare anno/peso senza scaricare tutto
                status, headers = head_for_meta(doc_url)
                if status:
                    ctype = (headers.get("Content-Type") or "").lower()
                    clen  = int(headers.get("Content-Length", "0") or "0")
                    y_hdr = _year_from_last_modified(headers)
                    y_url = _year_from_url(doc_url)
                    year = y_hdr if y_hdr is not None else y_url

                # 2) filtro anno
                if not _allowed_year(year):
                    skipped += 1
                    write_jsonl(index_out, {
                        "id": hashlib.md5(doc_url.encode()).hexdigest(),
                        "url": doc_url,
                        "referer": referer,
                        "saved_path": "",
                        "filename_original": filename_original,
                        "file_ext": ext,
                        "content_type": ctype,
                        "status_code": status,
                        "content_length": clen,
                        "year": year,
                        "downloaded": False,
                        "download_timestamp": int(time.time()),
                        "notes": "SKIPPED: filtro anno"
                    })
                    pbar.update(1)
                    time.sleep(SLEEP_BETWEEN)
                    continue

                # 3) GET effettivo del contenuto
                r = SESSION.get(doc_url, timeout=HTTP_TIMEOUT, allow_redirects=True)
                status = r.status_code
                if not r.ok or not r.content:
                    skipped += 1
                    write_jsonl(index_out, {
                        "id": hashlib.md5(doc_url.encode()).hexdigest(),
                        "url": doc_url,
                        "referer": referer,
                        "saved_path": "",
                        "filename_original": filename_original,
                        "file_ext": ext,
                        "content_type": ctype,
                        "status_code": status,
                        "content_length": clen,
                        "year": year,
                        "downloaded": False,
                        "download_timestamp": int(time.time()),
                        "notes": "Errore download"
                    })
                    pbar.update(1)
                    time.sleep(SLEEP_BETWEEN)
                    continue

                ctype = (r.headers.get("Content-Type") or ctype).lower()
                clen  = int(r.headers.get("Content-Length", str(len(r.content))) or str(len(r.content)))
                content = r.content
                sha = sha256_bytes(content)

                # path finale: <sha8>__<filename_sicuro>
                final_name = f"{sha[:8]}__{filename_original}"
                dest = DOCUMENTS_DIR / final_name

                if dest.exists() and dest.stat().st_size == len(content):
                    saved_path = str(dest.relative_to(DATA_DIR))
                else:
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    tmp.write_bytes(content)
                    tmp.replace(dest)
                    saved_path = str(dest.relative_to(DATA_DIR))
                    saved += 1

                write_jsonl(index_out, {
                    "id": sha or hashlib.md5(doc_url.encode()).hexdigest(),
                    "url": doc_url,
                    "referer": referer,
                    "saved_path": saved_path,             # relativo a data/
                    "filename_original": filename_original,
                    "file_ext": ext,
                    "content_type": ctype,
                    "status_code": status,
                    "content_length": clen,
                    "year": year,
                    "downloaded": True,
                    "download_timestamp": int(time.time()),
                })

            except Exception:
                skipped += 1
                write_jsonl(index_out, {
                    "id": hashlib.md5(doc_url.encode()).hexdigest(),
                    "url": doc_url,
                    "referer": referer,
                    "saved_path": "",
                    "filename_original": filename_original,
                    "file_ext": ext,
                    "content_type": ctype,
                    "status_code": status,
                    "content_length": clen,
                    "year": year,
                    "downloaded": False,
                    "download_timestamp": int(time.time()),
                    "notes": "Eccezione generica"
                })
            finally:
                pbar.update(1)
                time.sleep(SLEEP_BETWEEN)

        tqdm.write(f">> Documenti salvati: {saved}, saltati: {skipped}")
        tqdm.write(f">> Indice: {DOCS_INDEX_FILE}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if DEBUG_VERBOSE and not SCRAPER_QUIET:
        print(">> Avvio Document Downloader (crawler + filtro + download)")
        print(f">> Estensioni: {sorted(DOCS_EXTS)}")
        print(f">> Output index: {DOCS_INDEX_FILE}")
        print(f">> Destinazione: {DOCUMENTS_DIR}")
        print(f">> Policy anno: DOCS_YEAR_POLICY={DOCS_YEAR_POLICY}, DOCS_ACCEPT_UNDATED={DOCS_ACCEPT_UNDATED}, DOCS_MIN_YEAR={DOCS_MIN_YEAR or '-'}")
        if DOCS_DRY_RUN:
            print(">> Modalità DOCS_DRY_RUN=true (analisi HEAD, nessun download).")

    # Disabilita progress bar se quiet
    if SCRAPER_QUIET:
        try:
            from tqdm import tqdm as _t
            _t.disable = True  # type: ignore[attr-defined]
        except Exception:
            pass

    links = discover_document_links()
    if not links:
        tqdm.write(">> Nessun link a documenti trovato.")
        DOCS_INDEX_FILE.write_text("", encoding="utf-8")
        return


    if DOCS_DRY_RUN:
    # Solo analisi: nessun download, nessun file su disco
        analyze_document_links(links)
        return  # <-- STOP qui, mai scendere sotto

    # Download reale
    download_documents(links)


if __name__ == "__main__":
    main()