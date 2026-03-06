# Freud KB – Guida completa (Zero Cost RAG: Ollama + ChromaDB + Flowise)

Questa guida ti accompagna **passo-passo** dalla macchina “vuota” fino al chatbot **RAG** funzionante, usando **solo strumenti gratuiti**: **Ollama** (modelli locali), **ChromaDB** (vector store locale), **Flowise** (orchestrazione no/low-code).

---

## 0) Prerequisiti e obiettivi
- ✅ Windows 10/11
- ✅ VS Code (consigliato)
- ✅ Nessun costo cloud (NO Azure AI Search)
- 🎯 Obiettivo: FAQ/operativo scolastico con **citazioni obbligatorie**, **privacy**, **no-hallucination**

---

## 1) Installazione Python + pip (solo se non già fatto)
```powershell
winget install --id Python.Python.3.12 -e
``
-
python --version
pip --version
winget install Ollama.Ollama -e
ollama --version
ollama pull nomic-embed-text
ollama pull mistral

struttura 

freud-kb/
  data/
    s-freud.knowledge-ingestion.v1.1.enriched.json   # lista URL ufficiali istitutofreud.it
  build/
    pages/                                           # html/txt/pdf estratti
    chroma_freud/                                    # indice Chroma (si crea dopo)
  scripts/
    00_clean_freud_kb.py                             # pulizia dei file nelle cartelle contenenti le info de db KB
    01_scrape_freud.py                               # entra nel sito scansione dinamicamente menu e sottomenu visita e pulisce le pagine html 
                                                     # estrae il testo utile 
    02_download_documents.py                         # crawler:     salva i documenti con nome <sha8>__nomefile.ext                  
    03_index_chroma.py
  systemPromptLLM.txt
  flow.json
  tests_pcto_freud.csv

  pip install requests beautifulsoup4 lxml tqdm chromadb pypdf pdfplumber

AVVIA
  npx flowise start

  
cd freud-kb
python chat_ragNew.py

REPL Python 