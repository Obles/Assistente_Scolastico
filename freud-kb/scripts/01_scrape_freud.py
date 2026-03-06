# scripts/01_scrape_freud.py
# Scraper A+ (Dynamic Menu & Submenu Discovery) — Produce:
#   (A) catalogo_freud.jsonl  → mappa del sito (metadata-only)
#   (B) freud_pages.jsonl     → contenuti puliti per il RAG
#
# Caratteristiche:
# - Scoperta DINAMICA di menu e sotto‑menu dalla homepage (header/nav/footer/mega‑menu)
# - Integrazione con sitemap (se presente), filtrata dai prefissi dinamici
# - Anti‑loop: blocco fragment, querystring opzionale (da .env), normalizzazione URL,
#   dedup solido su URL canonicali, filtri “rumore”, limiti MAX_PAGES / MAX_DEPTH
# - Barra di avanzamento (tqdm) e modalità DRY‑RUN per diagnostica
# - Percorsi centralizzati via common_config.py (root = freud-kb/)
#
# Esecuzione:
#   python scripts/01_scrape_freud.py
#
# Output:
#   build/<timestamp>/catalogo_freud.jsonl
#   build/<timestamp>/freud_pages.jsonl
#   build/pages/*.html|*.txt|*.pdf
#   (diagnostica) build/<timestamp>/menu_urls.txt  [se DEBUG_VERBOSE o DRY‑RUN]

import os
import re
import io
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin, urlunparse, parse_qsl, urlencode
from typing import Optional, Set, List, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# PDF (opzionale)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# Config centralizzata (root = freud-kb/)
from common_config import (
    BUILD_DIR, PAGES_DIR,
    HTTP_TIMEOUT, MAX_RETRIES, BACKOFF_SEC,
    get_catalogo_path, get_rag_pages_path
)

# ---------------------------------------------------------------------
# PARAMETRI DA .env
# ---------------------------------------------------------------------
DOMAIN = "istitutofreud.it"
HOME_URL = "https://www.istitutofreud.it/"

MAX_DEPTH     = int(os.getenv("SCRAPER_MAX_DEPTH", "3"))
SLEEP_BETWEEN = float(os.getenv("SCRAPER_SLEEP_SEC", "0.25"))
MAX_PAGES     = int(os.getenv("SCRAPER_MAX_PAGES", "1200"))

# Blocchi
BLOCK_QUERYSTRING = os.getenv("SCRAPER_ALLOW_QUERYSTRING", "false").lower() not in ("1", "true", "yes")
BLOCK_FRAGMENT    = True

# Modalità diagnostica
DEBUG_VERBOSE    = os.getenv("SCRAPER_DEBUG_VERBOSE", "false").lower() in ("1","true","yes")
DEBUG_SAVE_EMPTY = os.getenv("SCRAPER_DEBUG_SAVE_EMPTY", "false").lower() in ("1","true","yes")
SCRAPER_QUIET    = os.getenv("SCRAPER_QUIET", "false").lower() in ("1", "true", "yes")
SCRAPER_DRY_RUN  = os.getenv("SCRAPER_DRY_RUN", "false").lower() in ("1", "true", "yes")

# Sitemap candidate
SITEMAP_CANDIDATES = [
    "https://www.istitutofreud.it/sitemap.xml",
    "https://www.istitutofreud.it/sitemap_index.xml",
    "https://www.istitutofreud.it/sitemap-index.xml",
]

UA = {"User-Agent": "Mozilla/5.0 (FreudKB-PCTO/dynamic-menu-scraper)"}

# Directory output
BUILD_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
CATALOGO_FILE = get_catalogo_path()
RAG_FILE      = get_rag_pages_path()

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
    r"/tag/", r"/categoria/", r"/category/", r"/author/", r"/feed/?$",
    r"/wp-json", r"/jsonapi", r"/search", r"/\d{4}/\d{2}/\d{2}/",
    r"/page/\d+/?$", r"/\?p=\d+",
]
DISALLOWED_RE = [re.compile(rx, re.I) for rx in DISALLOWED_REGEXES]

def _host_in_domain(netloc: str) -> bool:
    host = (netloc or "").lower()
    # consenti sottodomini di istitutofreud.it (www., ecc.)
    return host.endswith(DOMAIN) or host == DOMAIN

def is_directory_like(path: str) -> bool:
    leaf = (path or "/").split("/")[-1]
    return "." not in leaf

def strip_tracking(q_items):
    out = []
    for k, v in q_items:
        kl = k.lower()
        if kl in TRACKING_PARAMS:
            continue
        if any(kl.startswith(pref) for pref in TRACKING_PREFIXES):
            continue
        out.append((k, v))
    return out

def normalize_url(base: str, href: str) -> Optional[str]:
    """
    Assolutizza, forza dominio, rimuove fragment e (se impostato) query, compattando slash e trailing su directory-like.
    """
    if not href:
        return None
    try:
        abs_url = urljoin(base, href)
        u = urlparse(abs_url)
        if u.scheme not in ("http", "https"):
            return None
        if not _host_in_domain(u.netloc):
            return None

        # Query/fragment
        query = ""
        if not BLOCK_QUERYSTRING:
            query = urlencode(strip_tracking(parse_qsl(u.query, keep_blank_values=False)))
        fragment = ""  # rimosso sempre (anti-loop)
        if not BLOCK_FRAGMENT:
            fragment = u.fragment or ""

        # Path compattato + trailing slash standard su directory-like
        path = re.sub(r"/{2,}", "/", u.path or "/")
        if is_directory_like(path) and not path.endswith("/"):
            path = path + "/"

        new = u._replace(query=query, fragment=fragment, path=path)
        return urlunparse(new)
    except Exception:
        return None

def canonicalize_from_html(url: str, html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        link = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
        if link and link.get("href"):
            cand = normalize_url(url, link["href"])
            if cand:
                return cand
    except Exception:
        pass
    return url

def is_disallowed_by_pattern(path: str) -> bool:
    low = (path or "").lower()
    for rx in DISALLOWED_RE:
        if rx.search(low):
            return True
    return False

# ---------------------------------------------------------------------
# Estrazione HTML/PDF
# ---------------------------------------------------------------------
def clean_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    DROP_TAGS = {"nav", "header", "footer", "aside", "form", "noscript", "script", "style"}
    for tag in soup.find_all(DROP_TAGS):
        try:
            tag.decompose()
        except Exception:
            pass

    DROP_NOISE = re.compile(r"(cookie|banner|gdpr|newsletter|subscribe|popup|advert|modal)", re.I)
    # Gestione sicura di el.attrs che può essere None
    for el in list(soup.find_all(True)):
        try:
            attrs = getattr(el, "attrs", {}) or {}
            sid = attrs.get("id", "")
            scls = attrs.get("class", "")
            id_text  = " ".join(sid) if isinstance(sid, list) else (sid or "")
            cls_text = " ".join(scls) if isinstance(scls, list) else (scls or "")
            if (id_text and DROP_NOISE.search(id_text)) or (cls_text and DROP_NOISE.search(cls_text)):
                el.decompose()
        except Exception:
            continue

    return soup

def extract_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    return ""

# ---------------------------------------------------------------------
# EXTRACTOR (soluzione C): SEZIONE + contatti condizionali
# ---------------------------------------------------------------------
SECTION_RULES = [
    # (predicate, section_label, include_contacts)
    (lambda url, t, h: any(k in url for k in ["/contatti", "/contattaci", "/contatto", "/contattare"]), "Contatti Segreteria", True),
    (lambda url, t, h: "organigramma" in url or re.search(r"\borganigramma\b", (t + " " + h), re.I), "Organigramma", True),
    (lambda url, t, h: any(k in url for k in ["/servizi", "/utilita/faq"]), "Servizi/FAQ", False),
    (lambda url, t, h: any(k in url for k in ["/bacheca", "/news", "/editoriale"]), "News", False),
    (lambda url, t, h: any(k in url for k in ["/offerta-formativa", "/indirizzi"]), "Offerta Formativa", False),
]

def _guess_url_from_soup(soup: BeautifulSoup) -> str:
    can = soup.find("link", rel=lambda v: v and v.lower()=="canonical")
    if can and can.get("href"):
        return (can.get("href") or "").strip().lower()
    meta_og = soup.find("meta", property="og:url")
    if meta_og and meta_og.get("content"):
        return (meta_og.get("content") or "").strip().lower()
    return ""

def _detect_section(url: str, title_text: str, h1_text: str):
    url_l, t, h = (url or "").lower(), (title_text or ""), (h1_text or "")
    for pred, label, include in SECTION_RULES:
        try:
            if pred(url_l, t, h):
                return label, include
        except Exception:
            continue
    if re.search(r"\bcontatti?|segreteria\b", (t + " " + h), re.I):
        return "Contatti Segreteria", True
    return "Generico", False

def extract_text_and_section(soup: BeautifulSoup, url: Optional[str] = None) -> tuple[str, str]:
    """
    Ritorna (text, section_label).
    - Prefisso "SEZIONE: …" sempre
    - Contatti in testa SOLO per Contatti/Organigramma/Segreteria
    """
    body = soup.body or soup

    title_text = (soup.title.get_text(" ", strip=True) if soup.title else "")
    h1 = body.find("h1")
    h1_text = h1.get_text(" ", strip=True) if h1 else ""

    if not url:
        url = _guess_url_from_soup(soup).lower()

    section_label, include_contacts = _detect_section(url, title_text, h1_text)

    # testo base
    parts = []
    for el in body.find_all(["h1","h2","h3","h4","p","li"], recursive=True):
        t = el.get_text(" ", strip=True)
        if t:
            parts.append(t)
    main_text = re.sub(r"\s+", " ", " ".join(parts)).strip()

    # contatti (estrazione completa, uso condizionale)
    contacts = []
    for a in body.select('a[href^="tel:"]'):
        href = (a.get("href") or "")[4:]
        tel = re.sub(r"[^0-9+]", "", href)
        if tel and f"Telefono: {tel}" not in contacts:
            contacts.append(f"Telefono: {tel}")
    for a in body.select('a[href*="fax"]'):
        href = a.get("href") or ""
        fax = re.sub(r"[^0-9+]", "", href)
        if fax and f"Fax: {fax}" not in contacts:
            contacts.append(f"Fax: {fax}")
    for a in body.select('a[href^="mailto:"]'):
        href = (a.get("href") or "")[7:]
        email = href.split("?")[0]
        if email and f"Email: {email}" not in contacts:
            contacts.append(f"Email: {email}")
    for el in body.select('[itemprop="telephone"]'):
        val = el.get("content") or el.get_text(strip=True)
        tel = re.sub(r"[^0-9+]", "", val or "")
        if tel and f"Telefono: {tel}" not in contacts:
            contacts.append(f"Telefono: {tel}")
    for el in body.select('[itemprop="email"]'):
        val = el.get("content") or el.get_text(strip=True)
        email = (val or "").split("?")[0]
        if email and f"Email: {email}" not in contacts:
            contacts.append(f"Email: {email}")
    for a in body.select('a[href*="wa.me"], a[href*="api.whatsapp.com"]'):
        href = a.get("href") or ""
        wa = re.sub(r"[^0-9+]", "", href)
        if wa and f"WhatsApp: {wa}" not in contacts:
            contacts.append(f"WhatsApp: {wa}")

    # header + testo
    header_lines = [f"SEZIONE: {section_label}"]
    if include_contacts and contacts:
        header_lines.extend(contacts)
    text = ("\n".join(header_lines) + "\n" + main_text).strip()

    return text, section_label

# Wrapper di compatibilità: restituisce solo il testo
def extract_text(soup: BeautifulSoup, url: Optional[str] = None) -> str:
    text, _section = extract_text_and_section(soup, url=url)
    return text

def extract_pdf_text_bytes(pdf_bytes: bytes) -> str:
    if not pdfplumber:
        return ""
    try:
        buf = io.BytesIO(pdf_bytes)
        pages = []
        with pdfplumber.open(buf) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                if t:
                    pages.append(t)
        return re.sub(r"\s+", " ", " ".join(pages)).strip()
    except Exception:
        return ""

# ---------------------------------------------------------------------
# Menu + Sottomenu dinamici (homepage)
# ---------------------------------------------------------------------
def _collect_menu_links_from_container(base_url: str, container) -> Set[str]:
    urls: Set[str] = set()
    for a in container.find_all("a", href=True):
        n = normalize_url(base_url, a["href"])
        if n:
            urls.add(n)
    return urls

def discover_menu_and_submenus(home_url: str) -> Set[str]:
    """
    Estrae tutti i link di menu e sotto‑menu dalla homepage:
     - header, nav, footer
     - <ul>/<li> con classi tipiche: menu, nav, submenu, dropdown, mega
     - container con classi 'menu|nav|dropdown|submenu|mega'
    Ritorna un set di URL assoluti (normalizzati) nel dominio.
    """
    found: Set[str] = set()
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
            found |= _collect_menu_links_from_container(home_url, c)

        # includi sempre la home
        found.add(home_url)

        if DEBUG_VERBOSE or SCRAPER_DRY_RUN:
            all_urls = sorted(list(found))
            tqdm.write(f">> Menu/Sottomenu trovati: {len(all_urls)}")
            for u in all_urls:
                print(u)
            try:
                out_dir = Path(CATALOGO_FILE).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "menu_urls.txt").write_text("\n".join(all_urls), encoding="utf-8")
                tqdm.write(f">> Lista completa salvata in: {out_dir / 'menu_urls.txt'}")
            except Exception as e:
                tqdm.write(f">> Impossibile salvare lista menu: {e}")

        return set(found)
    except Exception:
        return {home_url}

def prefixes_from_urls(urls: Set[str]) -> Set[str]:
    """
    Dai link di menu/sottomenu ricava prefissi di path 'macro'.
    Esempi:
      /la-scuola/xyz  -> /la-scuola
      /bacheca/...    -> /bacheca
      /               -> /
    """
    prefs: Set[str] = set()
    for u in urls:
        p = urlparse(u).path or "/"
        segs = [s for s in p.split("/") if s]
        if not segs:
            prefs.add("/")
        else:
            prefs.add("/" + segs[0])
    # pulizia minima: rimuovi “figli” se un genitore più corto esiste
    cleaned: Set[str] = set()
    for p in sorted(prefs, key=len):
        if not any(p != q and p.startswith(q + "/") for q in prefs):
            cleaned.add(p)
    if not cleaned:
        cleaned.add("/")
    return cleaned

def is_allowed_by_prefix(path: str, allowed_prefixes: Set[str]) -> bool:
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
# Sitemap (filtrata dai prefissi dinamici)
# ---------------------------------------------------------------------
def read_sitemap_filtered(allowed_prefixes: Set[str]) -> List[str]:
    found: List[str] = []
    for sm in SITEMAP_CANDIDATES:
        try:
            r = SESSION.get(sm, timeout=HTTP_TIMEOUT)
            if r.ok and "xml" in (r.headers.get("Content-Type") or "").lower():
                # semplice estrazione delle <loc>
                locs = re.findall(r"<loc>(.*?)</loc>", r.text, flags=re.I | re.S)
                for loc in locs:
                    n = normalize_url(sm, loc.strip())
                    if not n:
                        continue
                    path = urlparse(n).path or "/"
                    if is_allowed_by_prefix(path, allowed_prefixes) and not is_disallowed_by_pattern(path):
                        found.append(n)
        except Exception:
            pass
    # dedup preservando ordine
    dedup = list(dict.fromkeys(found))
    if (DEBUG_VERBOSE or SCRAPER_DRY_RUN) and dedup:
        tqdm.write(f">> Sitemap URL utili: {len(dedup)} (post-filtri)")
    return dedup

# ---------------------------------------------------------------------
# Fetch + fallback con/senza slash (404)
# ---------------------------------------------------------------------
def fetch_with_fallback(url: str) -> Tuple[str, Optional[requests.Response]]:
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
# Crawler BFS
# ---------------------------------------------------------------------
def crawl() -> None:
    visited: Set[str]  = set()
    enqueued: Set[str] = set()

    # 1) Menu + sottomenu dinamici
    menu_urls = discover_menu_and_submenus(HOME_URL)
    allowed_prefixes = prefixes_from_urls(menu_urls)
    tqdm.write(f">> Prefissi ammessi (dinamici): {sorted(allowed_prefixes)}")

    if SCRAPER_DRY_RUN:
        tqdm.write(">> DRY-RUN attivo: nessun fetch dei contenuti, solo diagnostica di menu/sitemap.")
        _sitemap = read_sitemap_filtered(allowed_prefixes)
        tqdm.write(f">> DRY-RUN: sitemap utili trovate: {len(_sitemap)}")
        return

    # 2) Seeds iniziali = home + menu/sottomenu + sitemap filtrata
    seeds: List[str] = [HOME_URL]
    seeds += sorted(menu_urls)
    seeds += read_sitemap_filtered(allowed_prefixes)
    seeds = list(dict.fromkeys(seeds))

    disallowed_count = 0

    with open(CATALOGO_FILE, "w", encoding="utf-8") as cat_out, \
         open(RAG_FILE, "w", encoding="utf-8") as rag_out, \
         tqdm(total=MAX_PAGES, unit="page", dynamic_ncols=True, desc="Crawl Freud (dynamic menu)") as pbar:

        total = 0
        start_ts = time.time()

        # inizializza frontier con seeds ammessi
        frontier: List[Tuple[str,int]] = []
        for seed in seeds:
            s_norm = normalize_url(seed, seed)
            if not s_norm:
                continue
            sp = urlparse(s_norm).path or "/"
            if is_allowed_by_prefix(sp, allowed_prefixes) and not is_disallowed_by_pattern(sp):
                if s_norm not in enqueued:
                    frontier.append((s_norm, 0))
                    enqueued.add(s_norm)

        # BFS
        while frontier:
            url, depth = frontier.pop(0)

            if depth > MAX_DEPTH:
                continue
            if url in visited:
                continue
            if total >= MAX_PAGES:
                pbar.set_postfix_str(f"STOP MAX_PAGES={MAX_PAGES}")
                break

            fetch_url, resp = fetch_with_fallback(url)
            status = resp.status_code if resp else None
            ts_now = int(time.time())

            # Errore fetch → solo catalogo con errore
            if not resp or not resp.ok:
                write_jsonl(cat_out, {
                    "id": hashlib.md5(fetch_url.encode()).hexdigest(),
                    "title": "",
                    "url": fetch_url,
                    "category": "",
                    "keywords": [],
                    "description": "",
                    "source_type": "unknown",
                    "status_code": status,
                    "content_length": 0,
                    "crawl_timestamp": ts_now,
                    "notes": "Errore fetch / pagina non raggiungibile"
                })
                visited.add(fetch_url)  # evita retry inutili
                total += 1
                pbar.update(1)
                time.sleep(SLEEP_BETWEEN)
                continue

            ctype = (resp.headers.get("Content-Type") or "").lower()

            # --- PDF ---
            if "application/pdf" in ctype or fetch_url.lower().endswith(".pdf"):
                title_guess = Path(urlparse(fetch_url).path).name or "Documento PDF"
                text = extract_pdf_text_bytes(resp.content) if pdfplumber else ""

                write_jsonl(cat_out, {
                    "id": hashlib.md5(fetch_url.encode()).hexdigest(),
                    "title": title_guess,
                    "url": fetch_url,
                    "category": "",
                    "keywords": title_guess.lower().split(),
                    "description": "",
                    "source_type": "pdf",
                    "status_code": resp.status_code,
                    "content_length": len(text or ""),
                    "crawl_timestamp": ts_now,
                    "notes": "PDF rilevato"
                })

                if text and len(text) >= 50:
                    pid = hashlib.md5(fetch_url.encode()).hexdigest()
                    (PAGES_DIR / f"{pid}.pdf").write_bytes(resp.content)
                    (PAGES_DIR / f"{pid}.txt").write_text(text, encoding="utf-8")
                    write_jsonl(rag_out, {
                        "id": pid,
                        "url": fetch_url,
                        "title": title_guess,
                        "content": text,
                        "section": "Documento",     # metadato sezione per i PDF
                        "source_type": "pdf",
                        "timestamp": ts_now
                    })

                visited.add(fetch_url)
                total += 1
                pbar.update(1)
                pbar.set_postfix_str(f"depth≤{depth}")
                time.sleep(SLEEP_BETWEEN)
                continue

            # --- HTML ---
            html = resp.text or ""

            # canonical finale e normalizzazione
            final_url = canonicalize_from_html(fetch_url, html) or fetch_url
            n2 = normalize_url(final_url, final_url)
            if n2:
                final_url = n2

            if final_url in visited:
                total += 1
                pbar.update(1)
                time.sleep(SLEEP_BETWEEN)
                continue

            soup = clean_html(html)
            title = extract_title(soup)
            text, section = extract_text_and_section(soup, url=final_url)

            pid = hashlib.md5(final_url.encode()).hexdigest()
            write_jsonl(cat_out, {
                "id": pid,
                "title": title,
                "url": final_url,
                "category": "",
                "keywords": title.lower().split() if title else [],
                "description": "",
                "source_type": "html",
                "status_code": resp.status_code,
                "content_length": len(text or ""),
                "crawl_timestamp": ts_now,
                "notes": "Pagina HTML"
            })

            if text or DEBUG_SAVE_EMPTY:
                (PAGES_DIR / f"{pid}.html").write_text(html, encoding="utf-8", errors="ignore")
                if text:
                    (PAGES_DIR / f"{pid}.txt").write_text(text, encoding="utf-8")
                    write_jsonl(rag_out, {
                        "id": pid,
                        "url": final_url,
                        "title": title,
                        "content": text,
                        "section": section,          # << metadato chiave
                        "source_type": "html",
                        "timestamp": ts_now
                    })

            # marca visitato SOLO dopo canonicalizzazione → dedup stabile
            visited.add(final_url)
            total += 1
            pbar.update(1)
            pbar.set_postfix_str(f"depth≤{depth}")

            # Espansione link (BFS)
            if depth < MAX_DEPTH:
                for a in soup.find_all("a", href=True):
                    n = normalize_url(final_url, a["href"])
                    if not n:
                        continue
                    path_n = urlparse(n).path or "/"
                    if not is_allowed_by_prefix(path_n, allowed_prefixes):
                        disallowed_count += 1
                        continue
                    if is_disallowed_by_pattern(path_n):
                        disallowed_count += 1
                        continue
                    if n not in visited and n not in enqueued:
                        frontier.append((n, depth + 1))
                        enqueued.add(n)

            time.sleep(SLEEP_BETWEEN)

        elapsed = time.time() - start_ts
        tqdm.write(f">> COMPLETATO — Totale processati: {total} in {elapsed:.1f}s; scartati per filtri: {disallowed_count}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if DEBUG_VERBOSE and not SCRAPER_QUIET:
        print(">> Avvio Scraper Freud (dynamic, anti-loop, progress)")
        print(f">> Output Catalogo: {CATALOGO_FILE}")
        print(f">> Output RAG:      {RAG_FILE}")
        print(f">> Limiti: depth≤{MAX_DEPTH}, MAX_PAGES={MAX_PAGES}, BLOCK_QUERYSTRING={BLOCK_QUERYSTRING}")
        if SCRAPER_DRY_RUN:
            print(">> Modalità DRY-RUN attiva (no fetch contenuti, solo diagnostica menu/sitemap).")

    # Disabilita progress bar se quiet
    if SCRAPER_QUIET:
        try:
            from tqdm import tqdm as _t
            _t.disable = True  # type: ignore[attr-defined]
        except Exception:
            pass

    crawl()

if __name__ == "__main__":
    main()