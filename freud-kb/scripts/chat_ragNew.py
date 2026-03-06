# chat_ragNew.py
# Chat RAG (Chroma + Ollama) con:
# - Modalità Veloce (no LLM) / Completa (LLM)
# - Fallback A/B/C (rigoroso/semi/LLM judge)
# - Prompt istituzionale letto da systemPromptLLM.txt
# - Fonti obbligatorie “Titolo — URL”
# Avvio:  python chat_ragNew.py  →  http://127.0.0.1:7860
# Requisiti: gradio, requests, chromadb

import os
import re
import json
import time
import string
from pathlib import Path

import requests
import gradio as gr
from chromadb import PersistentClient

# ✅ Usa la configurazione centralizzata, come tutti gli altri script
from common_config import (
    PROJECT_DIR as ROOT_DIR,
    CHROMA_PATH as CHROMA_PATH_OBJ,
    CHROMA_DOCS_PATH as CHROMA_DOCS_PATH_OBJ,
)

# ============================ Config & Paths ================================

# Root progetto (freud-kb/)
PROJECT_DIR = ROOT_DIR

# Percorsi collection (assoluti, già normalizzati da common_config)
CHROMA_PATH = str(Path(CHROMA_PATH_OBJ).resolve())
CHROMA_DOCS_PATH = str(Path(CHROMA_DOCS_PATH_OBJ).resolve())

# Nomi delle collection (override da .env se presenti)
COLLECTION = os.getenv("COLLECTION", "freud_kb")
COLLECTION_DOCS = os.getenv("COLLECTION_DOCS", "freud_docs")

# Ollama
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", os.getenv("OLLAMA_URL", "http://localhost:11434/api/embeddings"))
OLLAMA_CHAT_URL  = os.getenv("OLLAMA_CHAT_URL",  "http://localhost:11434/api/chat")

# Modelli CPU-friendly
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # 768-dim
LLM_MODEL   = os.getenv("LLM_MODEL",   "mistral")

# Parametri LLM conservativi
LLM_OPTIONS = {
    "num_ctx": int(os.getenv("LLM_NUM_CTX", "768")),
    "num_predict": int(os.getenv("LLM_NUM_PREDICT", "120")),
    # "temperature": 0.2,
    # "top_p": 0.9,
}

# Limiti per il contesto passato all’LLM
# TOP_K_DEFAULT     = int(os.getenv("TOP_K", "3"))  # default 1 snippet (slider UI)


TOP_K_DEFAULT = int(os.getenv("TOP_K_GLOBAL", os.getenv("TOP_K", "3")))
TOP_K_HTML_ENV = int(os.getenv("TOP_K_HTML", "3"))
TOP_K_DOCS_ENV = int(os.getenv("TOP_K_DOCS", "3"))


MAX_CHARS_PER_DOC = int(os.getenv("MAX_CHARS_PER_DOC", "800"))
SEP               = "\n\n---\n\n"

# Prompt istituzionale da file esterno
SYSTEM_PROMPT_PATH = Path(os.getenv("SYSTEM_PROMPT_PATH", PROJECT_DIR / "systemPromptLLM.txt"))

# Fallback MUST (NO-ANSWER istituzionale)
FALLBACK_TEXT = (
    "L’informazione richiesta non risulta presente nei documenti e nelle pagine ufficiali attualmente disponibili. "
    "Per una conferma aggiornata, si invita a contattare la Segreteria.\n[CONTATTI SEGRETERIA]"
)

# ============================ Utils & Healthcheck ===========================

def _healthcheck_ollama(max_wait_sec: int = 8) -> bool:
    url = "http://localhost:11434/api/tags"
    t0 = time.time()
    while time.time() - t0 < max_wait_sec:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return True
        except Exception:
            time.sleep(0.5)
    return False

def _clean_space(s: str) -> str:
    return " ".join((s or "").split())

_MIN_IT_STOP = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","con","su","per","tra","fra",
    "e","o","che","come","del","della","dell","dei","degli","delle","al","allo","alla","ai","agli",
    "alle","dal","dallo","dalla","dai","dagli","dalle","nel","nello","nella","nei","negli","nelle",
    "col","coi","sul","sullo","sulla","sui","sugli","sulle","sono","è","era","quelle","questo","questa",
    "quello","quelle","queste","questi","dei","delle","dai","dal","dallo","dalla","alla","alle", "istituto","freud","scolastiche"
}

def _tokens(s: str):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in s.split() if t and t not in _MIN_IT_STOP]

# ============================ Init Chroma ===================================

def _open_collection():
    chroma_path_abs = CHROMA_PATH  # già assoluto/normalizzato
    try:
        client = PersistentClient(path=chroma_path_abs)
        col = client.get_collection(COLLECTION)
        return col
    except Exception as e:
        raise SystemExit(
            f"[ERRORE] Impossibile aprire la collection Chroma.\n"
            f"Percorso: {chroma_path_abs}\nCollection: {COLLECTION}\nDettagli: {e}"
        )

def _open_collection_docs():
    docs_path_abs = CHROMA_DOCS_PATH  # già assoluto/normalizzato
    try:
        client = PersistentClient(path=docs_path_abs)
        return client.get_collection(COLLECTION_DOCS)
    except Exception:
        return None

col = _open_collection()            # HTML
col_docs = _open_collection_docs()  # DOCS (può essere None)

# ============================ Embeddings ====================================

# def _try_parse_embedding(js):
#     if not isinstance(js, dict):
#         return None
#     emb = js.get("embedding")
#     if isinstance(emb, list) and emb:
#         return emb
#     data = js.get("data")
#     if isinstance(data, list) and data and isinstance(data[0], dict):
#         emb = data[0].get("embedding")
#         if isinstance(emb, list) and emb:
#             return emb
#     return None

def _try_parse_embedding(js):
    if not isinstance(js, dict):
        return None
    # Caso 1: {"embedding":[...]}
    emb = js.get("embedding")
    if isinstance(emb, list) and emb:
        return emb
    # Caso 2: {"data":[{"embedding":[...]}]}
    data = js.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        emb = data[0].get("embedding")
        if isinstance(emb, list) and emb:
            return emb
    # Caso 3: Ollama moderno -> {"embeddings":[ [..] ]} anche per input singolo
    embs = js.get("embeddings")
    if isinstance(embs, list) and embs and isinstance(embs[0], list):
        return embs[0]
    return None

def embed(text: str):
    payloads = [
        {"model": EMBED_MODEL, "prompt": text},       # 1) stile vecchio
        {"model": EMBED_MODEL, "input": text},        # 2) stile input singolo
        {"model": EMBED_MODEL, "prompt": [text]},     # 3) prompt array
        {"model": EMBED_MODEL, "input": [text]},      # 4) input array
    ]
    last_json = None
    for p in payloads:
        try:
            r = requests.post(OLLAMA_EMBED_URL, json=p, timeout=60)
            r.raise_for_status()
            js = r.json()
            emb = _try_parse_embedding(js)
            if emb:
                return emb
            last_json = js
        except requests.Timeout:
            raise RuntimeError("Timeout embeddings: Ollama impiega troppo tempo a rispondere.")
        except Exception:
            pass
    raise RuntimeError(f"Nessun embedding da Ollama (provate più varianti). Ultima risposta: {last_json}")

# ============================ LLM (chat) ====================================

def _parse_ndjson_concat(text: str) -> str:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            msg = (obj.get("message") or {}).get("content")
            if isinstance(msg, str):
                out.append(msg)
        except Exception:
            pass
    return "".join(out).strip()

def _load_system_prompt() -> str:
    try:
        p = SYSTEM_PROMPT_PATH
        if not p.exists():
            return (
                "Lei è un assistente istituzionale della scuola. "
                "Risponda solo in base alle fonti del knowledge base; in caso di assenza di fonti, usi il NO‑ANSWER istituzionale."
            )
        return p.read_text(encoding="utf-8")
    except Exception:
        return (
            "Lei è un assistente istituzionale della scuola. "
            "Risponda solo in base alle fonti del knowledge base; in caso di assenza di fonti, usi il NO‑ANSWER istituzionale."
        )

def llm_chat(system_prompt: str, user_prompt: str, options: dict | None = None) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": options or LLM_OPTIONS,
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_CHAT_URL, json=payload, stream=False, timeout=300)
        r.raise_for_status()
        # Ollama può restituire JSON standard o NDJSON
        try:
            js = r.json()
            if isinstance(js, dict):
                msg = (js.get("message") or {}).get("content")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
                if isinstance(js.get("response"), str) and js["response"].strip():
                    return js["response"].strip()
        except Exception:
            text = _parse_ndjson_concat(r.text)
            if text:
                return text
        return r.text.strip()
    except requests.Timeout:
        return ("[Avviso] Il modello ha impiegato troppo tempo a rispondere. "
                "Riprovi o usi un modello più leggero (es. 'llama3.2:3b-instruct').")
    except Exception as e:
        return f"[Errore LLM] {e}"

# ============================ Prompt & Fonti ================================

PROMPT_TMPL = (
    "CONTENUTO (estratti dai documenti ufficiali):\n{context}\n\n"
    "DOMANDA:\n{question}\n\n"
    "Istruzioni operative:\n"
    "- Risponda in modo breve, formale e istituzionale, SOLO se l’informazione è presente nel CONTENUTO.\n"
    "- Concluda con la sezione “Fonti”. In assenza di copertura, usi il NO‑ANSWER istituzionale con [CONTATTI SEGRETERIA].\n"
)

def ensure_fonti_section(reply: str, fonti_labels: list[str]) -> str:
    if not fonti_labels:
        return reply.strip()
    lower = reply.lower()
    if "fonti:" not in lower:
        suffix = "\n\nFonti:\n" + "\n".join(f"- {f}" for f in fonti_labels)
        return (reply.strip() + suffix).strip()
    return (reply.strip() + "\n" + "\n".join(f"- {f}" for f in fonti_labels)).strip()

def build_context_from_fused(results, clip_chars: int):
    parts, fonti = [], []
    for origin, text, meta, dist in results:
        snippet = _clean_space(text)[:clip_chars]
        if snippet:
            parts.append(snippet)
        url = (meta or {}).get("url")
        title = (meta or {}).get("title") or ""
        label_base = f"{title} — {url}" if url else (title or "")
        label = f"[{origin.upper()}] {label_base}" if label_base else f"[{origin.upper()}]"
        if label and label not in fonti:
            fonti.append(label)
    ctx = SEP.join(parts) if parts else ""
    return ctx, fonti

def quick_answer(ctx: str, fonti_labels: list[str]) -> str:
    base = "### Estratti pertinenti\n\n"
    base += (ctx if ctx else "Nessun estratto pertinente disponibile.")
    if ctx.strip():
        base = ensure_fonti_section(base, fonti_labels)
    return base

# ============================ Doppia Ricerca (HTML + DOCS) ==================

def _query_both_collections(q_emb, k_html: int, k_docs: int, k_global: int):
    results = []

    # HTML
    try:
        # res_html = col.query(query_embeddings=[q_emb], n_results=int(max(1, k_html)))
        
        res_html = col.query(
            query_embeddings=[q_emb],
            n_results=int(max(1, k_html)),
            include=["documents", "metadatas", "distances"]
        )
        docs  = (res_html.get("documents") or [[]])[0]
        metas = (res_html.get("metadatas") or [[]])[0]
        dists = (res_html.get("distances") or [[]])[0] or []
        for i, d in enumerate(docs):
            if not d:
                continue
            m = metas[i] if i < len(metas) else {}
            dist = float(dists[i]) if i < len(dists) else 0.0
            results.append(("html", d, m, dist))
    except Exception:
        pass

    # DOCS (solo se disponibile)
    if col_docs is not None and k_docs > 0:
        try:
            # res_docs = col_docs.query(query_embeddings=[q_emb], n_results=int(k_docs))
            
            res_docs = col_docs.query(
                query_embeddings=[q_emb],
                n_results=int(k_docs),
                include=["documents", "metadatas", "distances"]
            )
            dd   = (res_docs.get("documents") or [[]])[0]
            mm   = (res_docs.get("metadatas") or [[]])[0]
            ddis = (res_docs.get("distances") or [[]])[0] or []
            for i, d in enumerate(dd):
                if not d:
                    continue
                m = mm[i] if i < len(mm) else {}
                dist = float(ddis[i]) if i < len(ddis) else 0.0
                results.append(("docs", d, m, dist))
        except Exception:
            pass

    if not results:
        return []
    results.sort(key=lambda x: x[3])  # distanza crescente
    return results[:max(1, int(k_global))]

# ============================ Fallback Validators ============================

def needs_fallback_A_rigorous(question: str, context: str, fused_results=None) -> bool:
    """
    Fallback rigoroso basato sulla qualità della retrieval:
    - se non c’è contesto → fallback
    - se non ho risultati → fallback
    - se la distanza del miglior risultato supera una soglia → fallback
    """
    if not context.strip():
        return True
    if not fused_results:
        return True

    # fused_results: lista di tuple (origin, text, meta, distance) ordinata per distanza crescente
    best_distance = fused_results[0][3]

    # Soglia prudente (cosine distance in Chroma: più piccolo = meglio).
    # 0.35–0.40 è un intervallo comune; puoi affinarla.
    # THRESHOLD = 0.38
    THRESHOLD = float(os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.45"))
    return best_distance > THRESHOLD

def needs_fallback_B_semi(question: str, context: str) -> bool:
    q_tokens = _tokens(question)
    ctx_low = context.lower()
    if not ctx_low.strip():
        return True
    for tok in q_tokens:
        if tok in ctx_low:
            return False
    return True

def needs_fallback_C_llm(question: str, context: str) -> bool:
    if not context.strip():
        return True
    prompt = (
        "Contesto:\n"
        f"{context}\n\n"
        "Domanda:\n"
        f"{question}\n\n"
        "Rispondi SOLO con SI o NO: "
        "Il contesto qui sopra contiene informazioni sufficienti per rispondere correttamente alla domanda?"
    )
    try:
        resp = llm_chat("", prompt, options={"num_ctx":512, "num_predict":4})
        answer = (resp or "").strip().lower()
        if answer.startswith("s"):
            return False
        return True
    except Exception:
        return True

# ============================ Pipeline =====================================

def pipeline(message: str, use_llm: bool, top_k: int, policy: str):
    if not _healthcheck_ollama():
        return ("[Ollama non risponde su http://localhost:11434]\n"
                "- Avviare Ollama (Desktop/servizio)\n"
                "- 'ollama list' deve mostrare i modelli\n"
                "- Se serve: 'ollama pull mistral' e 'ollama pull nomic-embed-text'")

    try:
        q_emb = embed(message)
        # k_html = max(1, top_k // 2)
        # k_docs = max(0, top_k - k_html) if col_docs is not None else 0
        # k_global = max(1, top_k)
        
        # 🔁 Nuova strategia: prendo più candidati da entrambe le collection
        # e poi tengo i migliori top_k globali.
        # K_BOOST = max(3, top_k)  # almeno 3 HTML e 3 DOCS
        # k_html = K_BOOST
        # k_docs = K_BOOST if col_docs is not None else 0
        # k_global = max(1, top_k)
        
        k_html = max(3, TOP_K_HTML_ENV, top_k)
        k_docs = max(0, TOP_K_DOCS_ENV, top_k) if col_docs is not None else 0
        k_global = max(1, top_k)


        fused = _query_both_collections(q_emb, k_html, k_docs, k_global)
    except Exception as e:
        return f"[Errore] {e}"

    if fused:
        context, fonti_labels = build_context_from_fused(fused, clip_chars=MAX_CHARS_PER_DOC)
    else:
        context, fonti_labels = "", []

    policy_key = (policy or "").strip().upper()[:1]
    if   policy_key == "A":
        need_fb = needs_fallback_A_rigorous(message, context, fused_results=fused)
    elif policy_key == "B":
        need_fb = needs_fallback_B_semi(message, context)
    else:
        need_fb = needs_fallback_C_llm(message, context)

    if need_fb:
        fb = FALLBACK_TEXT
        if fonti_labels:
            fb += "\n\nFonti consultate (non contenenti la risposta):\n" + "\n".join(f"- {f}" for f in fonti_labels)
        return fb

    if not use_llm:
        return quick_answer(context, fonti_labels)

    system_prompt = _load_system_prompt()
    user_prompt   = PROMPT_TMPL.format(context=context, question=message)
    reply         = llm_chat(system_prompt, user_prompt, options=LLM_OPTIONS)
    return ensure_fonti_section(reply, fonti_labels)

# ============================ UI (Gradio 6.x) ==============================

with gr.Blocks(title="Freud-KB RAG (Chroma + Ollama)") as demo:
    gr.Markdown("# Freud-KB RAG — Chat locale (Chroma + Ollama)")
    gr.Markdown(
        "Scriva una domanda. Il sistema userà esclusivamente il database locale indicizzato per rispondere, "
        "in conformità alle regole istituzionali (SOLO FONTI)."
    )

    use_llm = gr.Checkbox(label="Usa LLM (risposta completa)", value=False)
    top_k   = gr.Slider(1, 3, value=TOP_K_DEFAULT, step=1, label="Top‑K documenti (contesto)")

    fallback_policy = gr.Radio(
        choices=[
            "A) Rigorosa (token domanda devono comparire nel contesto)",
            "B) Semi‑rigorosa (contesto minimo richiesto)",
            "C) LLM decide (micro‑verifica SI/NO)"
        ],
        label="Politica di fallback",
        value="A) Rigorosa (token domanda devono comparire nel contesto)"
    )

    chatbox   = gr.Chatbot(label="Chat", height=420)
    textbox   = gr.Textbox(lines=2, label="Domanda")
    send_btn  = gr.Button("Invia", variant="primary", interactive=False)
    clear_btn = gr.Button("Pulisci chat")

    def _toggle_send(txt):
        return gr.update(interactive=bool((txt or "").strip()))

    textbox.change(_toggle_send, inputs=textbox, outputs=send_btn)

    def _on_submit(user_msg, messages, use_llm_v, topk_v, policy_v):
        user_msg = (user_msg or "").strip()
        if not user_msg:
            return messages, gr.update(value="", interactive=False)
        messages = messages or []
        messages.append({"role": "user", "content": user_msg})
        reply = pipeline(user_msg, use_llm_v, topk_v, policy_v)
        messages.append({"role": "assistant", "content": reply})
        return messages, gr.update(value="", interactive=True)

    send_btn.click(
        _on_submit,
        inputs=[textbox, chatbox, use_llm, top_k, fallback_policy],
        outputs=[chatbox, textbox]
    )

    textbox.submit(
        _on_submit,
        inputs=[textbox, chatbox, use_llm, top_k, fallback_policy],
        outputs=[chatbox, textbox]
    )

    def _clear_chat():
        return [], gr.update(value="", interactive=False)

    clear_btn.click(_clear_chat, None, [chatbox, textbox])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)