# Checklist di Validazione Tecnica del RAG — FreudKB

File scaricabile: [RAG_checklist_validazione_tecnica.md](sandbox:/mnt/data/RAG_checklist_validazione_tecnica.md)

Versione documento: 2026-03-06  
Scopo: fornire una checklist **esaustiva, trasferibile in altre chat** e utilizzabile come base di audit tecnico del progetto RAG locale FreudKB, **senza modificare il codice**.  
Perimetro: scraper HTML, crawler documenti, indicizzazione Chroma, embeddings Ollama, chat RAG, prompt “SOLO FONTI”, orchestrazione pipeline, test tecnico e coerenza documentazione.

---

## 1. Executive summary

Lo sviluppo del RAG è **strutturalmente valido**, ma **non può essere considerato completamente validato** allo stato attuale.

L’architettura dichiarata è coerente:
- pipeline a step (`00` pulizia, `01` scraping HTML, `02` crawler documenti, `03` index HTML, `03b` index DOCS, chat);
- **due collection Chroma separate** per HTML e documenti;
- retrieval parallelo con **fusione dei risultati**;
- metrica **cosine**;
- fallback A/B/C;
- policy conversazionale **“SOLO FONTI”**.:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}

Tuttavia emergono criticità reali che impediscono di chiudere l’audit con esito “verde pieno”:
- il chunk overlap dichiarato non risulta effettivamente applicato nell’indexer HTML;
- il filtro anno dei documenti presenta una regex sospetta/errata;
- gli endpoint embeddings di Ollama risultano **non pienamente allineati** tra guide, `.env` e chat;
- il test tecnico copre la collection HTML, ma **non prova in modo completo la pipeline dual-collection usata in chat**;
- la chat richiede `gradio`, ma la dipendenza non risulta allineata nelle fonti documentali principali.:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}

---

## 2. Criteri di severità usati in questo audit

### Bloccante
Problema che può invalidare la qualità del retrieval, rompere l’avvio della pipeline o rendere non affidabile il comportamento del sistema rispetto all’architettura dichiarata.

### Alta
Problema che non rompe sempre il sistema, ma compromette in modo concreto affidabilità, copertura o riproducibilità.

### Media
Problema che introduce ambiguità operative, errori di setup o test incompleti, senza necessariamente impedire l’uso del sistema.

### Bassa
Problema di igiene tecnica o documentale, con impatto soprattutto manutentivo.

---

## 3. Stato architetturale atteso

La baseline corretta, secondo la documentazione più allineata al progetto corrente, è la seguente:

1. `00_clean_freud_kb.py` opzionale per ripartenza pulita.  
2. `01_scrape_freud.py` per scraping HTML e patch contatti.  
3. `02_download_documents.py` opzionale per crawler/download documenti.  
4. `03_index_freud.py` per indicizzazione HTML con Chroma cosine.  
5. `03b_index_documents.py` per indicizzazione documenti con Chroma cosine.  
6. `test_rag.py` come test tecnico.  
7. `chat_ragNew.py` come interfaccia Gradio e runtime della risposta RAG.:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}

Il modello operativo dichiarato è un RAG con:
- **collection HTML** (`build/chroma_freud`);
- **collection DOCS** (`build/chroma_freud_docs`);
- **ricerca parallela** e **fusione risultati**;
- soglia `RETRIEVAL_DISTANCE_THRESHOLD`;
- NO-ANSWER istituzionale in assenza di copertura documentale.:contentReference[oaicite:11]{index=11}:contentReference[oaicite:12]{index=12}

---

## 4. Checklist di validazione ordinata per severità

# A. BLOCCANTE

## A1. Verificare che il chunk overlap sia realmente applicato negli indexer

### Perché è critico
La documentazione dichiara chunking configurabile con `CHUNK_TARGET_CHARS`, `CHUNK_MIN_CHARS`, `CHUNK_OVERLAP_CHARS` e presenta l’overlap come parte della qualità del retrieval.:contentReference[oaicite:13]{index=13}:contentReference[oaicite:14]{index=14}

Nel codice dell’indexer HTML, però, la logica mostrata è:
- `stop = min(start + target, n)`
- poi `start = max(stop - overlap, stop)`:contentReference[oaicite:15]{index=15}

Questa espressione è logicamente sospetta perché `max(stop - overlap, stop)` restituisce sempre `stop`, quindi **l’overlap effettivo risulta nullo**.

### Cosa validare
- Confermare se lo stesso schema è presente anche nell’indexer documenti (`03b_index_documents.py`).
- Verificare, a livello di output indicizzato, se chunk consecutivi si sovrappongono davvero oppure no.
- Controllare se le query che dipendono da contenuto distribuito su bordo-chunk hanno recall inferiore al previsto.

### Evidenze da raccogliere
- Estratto del codice `chunk_text()` in HTML e DOCS.
- Esempio di due chunk consecutivi della stessa pagina/documento.
- Verifica che il secondo chunk inizi **prima** della fine del precedente; se inizia esattamente a `stop`, overlap assente.

### Esito atteso
- **PASS**: i chunk consecutivi condividono davvero una porzione di testo.  
- **FAIL**: i chunk sono contigui ma non sovrapposti.  
- **Rischio**: perdita di recall su informazioni spezzate tra chunk adiacenti.

### Stato attuale dell’audit
**FAIL probabile / da confermare sul 03b**.:contentReference[oaicite:16]{index=16}

---

## A2. Verificare la correttezza reale del filtro anno documenti

### Perché è critico
Il crawler documenti usa una policy anno (`DOCS_YEAR_POLICY`, `DOCS_ACCEPT_UNDATED`, `DOCS_MIN_YEAR`) per decidere se includere o escludere file.:contentReference[oaicite:17]{index=17}

Nel codice del downloader, la funzione `_year_from_url()` mostra regex del tipo:
- `re.search(r"/(20\\d{2})/", url)`
- `re.search(r"(20\\d{2})", fname)`:contentReference[oaicite:18]{index=18}

La presenza di `\\d` dentro raw string è fortemente sospetta: così si cerca letteralmente `\d` invece del pattern numerico `\d`. Se confermato, il filtro anno da URL/file name può fallire sistematicamente.

### Cosa validare
- Verificare se `_year_from_url()` riconosce davvero URL come `/2026/` o file tipo `circolare_2025.pdf`.
- Verificare quanti documenti risultano “undated” pur contenendo l’anno in URL o filename.
- Verificare il comportamento combinato con `Last-Modified` e `DOCS_ACCEPT_UNDATED=true`.

### Evidenze da raccogliere
- Campione di URL con anno esplicito.
- Output dry-run del document crawler.
- Numero di file ammessi grazie a `Last-Modified` e numero di file ammessi come “undated”.

### Esito atteso
- **PASS**: URL e filename con anno vengono classificati correttamente.  
- **FAIL**: la stima anno da URL/file non funziona e il filtro è alterato.

### Stato attuale dell’audit
**FAIL probabile**.:contentReference[oaicite:19]{index=19}

---

## A3. Verificare che il runtime chat usi un endpoint embeddings coerente e funzionante

### Perché è critico
La guida operativa aggiornata dichiara che per la chat l’endpoint embeddings deve essere `/api/embed`.:contentReference[oaicite:20]{index=20}:contentReference[oaicite:21]{index=21}

La documentazione principale più recente mostra nel template `.env` `OLLAMA_URL=http://localhost:11434/api/embed`. :contentReference[oaicite:22]{index=22}

Tuttavia:
- una parte della documentazione riporta ancora `/api/embeddings`;:contentReference[oaicite:23]{index=23}
- `chat_ragNew.py` usa come fallback di default `http://localhost:11434/api/embeddings`. :contentReference[oaicite:24]{index=24}

Questo disallineamento può produrre setup che “sembrano giusti” ma falliscono in modo intermittente o dipendente dall’ambiente.

### Cosa validare
- Verificare il valore effettivo di `OLLAMA_URL`/`OLLAMA_EMBED_URL` a runtime.
- Verificare che la chat generi embeddings non vuoti e coerenti con `nomic-embed-text`.
- Verificare che test, indexer e chat convergano sullo stesso endpoint effettivo.

### Evidenze da raccogliere
- `.env` in uso.
- Output runtime della chat in fase embedding.
- Eventuale healthcheck manuale verso endpoint embeddings.

### Esito atteso
- **PASS**: endpoint unico e funzionante in tutte le fasi.  
- **FAIL**: test/index/chat usano endpoint differenti o documentazione contraddittoria.

### Stato attuale dell’audit
**FAIL documentale / rischio alto runtime**.:contentReference[oaicite:25]{index=25}:contentReference[oaicite:26]{index=26}:contentReference[oaicite:27]{index=27}

---

# B. ALTA

## B1. Verificare che il test tecnico copra davvero la pipeline finale HTML + DOCS

### Perché è importante
La guida PCTO presenta `test_rag.py` come test tecnico obbligatorio e lo descrive come apertura della collection HTML, verifica metrica cosine, embedding a 768 dimensioni e query top-5 con distances.:contentReference[oaicite:28]{index=28}

Nel codice disponibile, il test:
- genera embedding query;
- interroga una collection Chroma;
- mostra top-5 e best distance.:contentReference[oaicite:29]{index=29}

Ma non risulta, dalle fonti consultate, una verifica completa della **fusione HTML + DOCS** che invece è centrale nell’architettura della chat.:contentReference[oaicite:30]{index=30}

### Cosa validare
- Verificare se `test_rag.py` apre solo `CHROMA_PATH/COLLECTION` o anche `CHROMA_DOCS_PATH/COLLECTION_DOCS`.
- Verificare se esiste una prova integrata della fusione ranking HTML+DOCS.
- Verificare se una query presente solo nei documenti viene coperta dal test attuale.

### Evidenze da raccogliere
- Output di `test_rag.py` con query “solo documenti”.
- Evidenza che il test tocchi oppure non tocchi la collection DOCS.
- Confronto tra esito test e comportamento reale della chat.

### Esito atteso
- **PASS**: il test riproduce il comportamento reale della chat su doppia collection.  
- **FAIL**: il test certifica solo il retrieval HTML.

### Stato attuale dell’audit
**FAIL parziale / copertura incompleta**.:contentReference[oaicite:31]{index=31}:contentReference[oaicite:32]{index=32}

---

## B2. Verificare la coerenza metrica cosine tra dichiarazione, collection e soglia di retrieval

### Perché è importante
Le guide indicano chiaramente che HTML e DOCS devono essere indicizzati con `hnsw:space='cosine'`, coerentemente con la soglia `RETRIEVAL_DISTANCE_THRESHOLD`.:contentReference[oaicite:33]{index=33}:contentReference[oaicite:34]{index=34}

Il test tecnico descrive che la metadata della collection deve contenere `{'hnsw:space': 'cosine'}`.:contentReference[oaicite:35]{index=35}

### Cosa validare
- Verificare la metadata reale di entrambe le collection: HTML e DOCS.
- Verificare che la soglia usata in chat sia interpretata come **cosine distance**.
- Verificare che non esistano collection residue create con metrica diversa e poi riusate per errore.

### Evidenze da raccogliere
- Metadata collection HTML.
- Metadata collection DOCS.
- Eventuali directory Chroma stale sotto `build/`.

### Esito atteso
- **PASS**: entrambe le collection usano cosine; la soglia è coerente.  
- **FAIL**: metrica disallineata o collection ereditate.

### Stato attuale dell’audit
**DA CONFERMARE OPERATIVAMENTE**; la documentazione è coerente, ma serve verifica diretta sulle collection buildate.:contentReference[oaicite:36]{index=36}:contentReference[oaicite:37]{index=37}

---

## B3. Verificare che le dipendenze dichiarate bastino davvero ad avviare chat e pipeline

### Perché è importante
La chat richiede esplicitamente `gradio`, `requests`, `chromadb`.:contentReference[oaicite:38]{index=38}

Le appendici `requirements.txt` nelle guide mostrano però solo:
- `beautifulsoup4`
- `lxml`
- `tqdm`
- `requests`
- `pdfplumber`
- `chromadb`
- `python-dotenv`:contentReference[oaicite:39]{index=39}:contentReference[oaicite:40]{index=40}

Questa divergenza significa che la pipeline di base può installarsi, ma la chat potrebbe non partire in ambiente pulito.

### Cosa validare
- Verificare il contenuto reale di `requirements.txt` nel repository attuale.
- Verificare avvio chat su venv nuovo e pulito.
- Verificare che nessuna dipendenza sia ereditata implicitamente da ambiente globale.

### Evidenze da raccogliere
- `pip install -r requirements.txt` in ambiente pulito.
- `python chat_ragNew.py` subito dopo installazione.
- Log di eventuale `ModuleNotFoundError`.

### Esito atteso
- **PASS**: requirements sufficienti per pipeline e chat.  
- **FAIL**: setup documentato non riproduce l’avvio completo.

### Stato attuale dell’audit
**FAIL probabile in setup pulito**.:contentReference[oaicite:41]{index=41}:contentReference[oaicite:42]{index=42}

---

# C. MEDIA

## C1. Verificare che lo scraper HTML preservi davvero i contatti chiave per le query amministrative

### Perché è importante
La guida PCTO evidenzia una patch specifica per includere `mailto:`, `tel:` e microdati, concatenando i contatti prima del testo pagina.:contentReference[oaicite:43]{index=43}:contentReference[oaicite:44]{index=44}

Questa parte è fondamentale per query come “telefono segreteria”, “email segreteria didattica”, “contatti”.

### Cosa validare
- Verificare che il testo estratto contenga davvero telefoni/email in chiaro.
- Verificare che, dopo scraping e reindex, query di contatto recuperino pagine pertinenti.
- Verificare che la patch non sia presente solo in documentazione ma anche nel codice effettivo usato.

### Evidenze da raccogliere
- Estratto JSONL di una pagina “Contatti”.
- Query di test su telefono, email, segreteria.
- Ranking top-k e distanza.

### Esito atteso
- **PASS**: dati di contatto presenti e recuperabili.  
- **FAIL**: la fonte esiste sul sito ma non è stata serializzata nel corpus.

### Stato attuale dell’audit
**VEROSIMILMENTE BUONO ma da testare sul corpus prodotto**.:contentReference[oaicite:45]{index=45}

---

## C2. Verificare che i documenti brevi non vengano scartati impropriamente

### Perché è importante
La guida principale segnala esplicitamente il rischio “documenti corti non trovati” e afferma che `03b` deve includere anche testi inferiori a `min_chars`.:contentReference[oaicite:46]{index=46}

### Cosa validare
- Verificare la logica del chunking/ingestion per documenti piccoli in `03b_index_documents.py`.
- Verificare se avvisi, moduli, file molto corti finiscono comunque indicizzati.
- Verificare presenza di record documentali con testo corto ma non vuoto.

### Evidenze da raccogliere
- Elenco documenti indicizzati con contenuto < `CHUNK_MIN_CHARS`.
- Query su file brevi noti.

### Esito atteso
- **PASS**: i documenti corti restano recuperabili.  
- **FAIL**: i documenti brevi vengono silenziosamente persi.

### Stato attuale dell’audit
**DA CONFERMARE**; la guida dichiara la patch, ma serve verifica sul codice e sui dati indicizzati.:contentReference[oaicite:47]{index=47}

---

## C3. Verificare che la policy “SOLO FONTI” sia rispettata in chat e fallback

### Perché è importante
Il prompt istituzionale richiede che ogni risposta sia basata esclusivamente sulle fonti fornite, verificabile e priva di contenuti inventati o inferenze non supportate.:contentReference[oaicite:48]{index=48}

La chat dichiara anche fallback A/B/C e NO-ANSWER istituzionale.:contentReference[oaicite:49]{index=49}:contentReference[oaicite:50]{index=50}

### Cosa validare
- Verificare che in assenza di copertura il sistema non “riempia i buchi”.
- Verificare che la modalità completa non introduca inferenze non presenti negli estratti.
- Verificare che la modalità veloce sia coerente con le fonti mostrate.

### Evidenze da raccogliere
- Test con query non coperte dal corpus.
- Test con query ambigue.
- Confronto tra risposta LLM e contesto passato al modello.

### Esito atteso
- **PASS**: in mancanza di fonti emerge NO-ANSWER, non allucinazione.  
- **FAIL**: risposta apparentemente plausibile ma non supportata.

### Stato attuale dell’audit
**DA CONFERMARE CON TEST FUNZIONALI MIRATI**.:contentReference[oaicite:51]{index=51}

---

## C4. Verificare la coerenza del runbook e dell’orchestrazione pipeline

### Perché è importante
La guida operativa indica una sequenza precisa di step e un orchestratore `run_pipeline.py` che deve poter partire da uno step e proseguire fino al termine.:contentReference[oaicite:52]{index=52}:contentReference[oaicite:53]{index=53}

### Cosa validare
- Verificare che i nomi script reali corrispondano ai nomi usati nel runbook.
- Verificare che gli step presenti nell’orchestratore corrispondano davvero ai file nel repository.
- Verificare che l’orchestratore completi la sequenza senza fermarsi su incongruenze di naming/path.

### Evidenze da raccogliere
- Output di `run_pipeline.py 00`.
- Mapping step → file → esistenza su disco.

### Esito atteso
- **PASS**: esecuzione sequenziale coerente con la guida.  
- **FAIL**: runbook e file reali divergono.

### Stato attuale dell’audit
**MEDIO RISCHIO DOCUMENTALE**; le guide correnti appaiono coerenti, ma esistono tracce di documentazione vecchia che possono confondere il runbook operativo.:contentReference[oaicite:54]{index=54}:contentReference[oaicite:55]{index=55}

---

# D. BASSA

## D1. Verificare ed etichettare chiaramente le fonti documentali obsolete

### Perché è importante
Esiste almeno una guida che parla ancora di script come `01_enrich_kb.py` e `02_index_chroma.py`, non coerenti con la pipeline attuale (`01_scrape_freud.py`, `03_index_freud.py`, `03b_index_documents.py`).:contentReference[oaicite:56]{index=56}:contentReference[oaicite:57]{index=57}

### Cosa validare
- Elencare quali documenti sono “correnti” e quali “archivio”.
- Evitare che guide obsolete vengano usate come fonte di verità tecnica.

### Esito atteso
- **PASS**: perimetro documentale ufficiale definito.  
- **FAIL**: più guide concorrenti con pipeline incompatibili.

### Stato attuale dell’audit
**FAIL documentale lieve**.:contentReference[oaicite:58]{index=58}

---

## D2. Verificare l’igiene dei parametri `.env` e il loro allineamento con la documentazione

### Perché è importante
Le guide mostrano template `.env` ricchi e coerenti con la pipeline attuale, inclusi `CHROMA_PATH`, `CHROMA_DOCS_PATH`, `COLLECTION_DOCS`, `TOP_K_*`, `CHUNK_*`, `DOCS_*`.:contentReference[oaicite:59]{index=59}:contentReference[oaicite:60]{index=60}

### Cosa validare
- Verificare che il `.env` reale abbia tutti i campi chiave valorizzati e senza duplicazioni concettuali ambigue.
- Verificare che eventuali override (`OLLAMA_EMBED_URL`) non introducano conflitti con `OLLAMA_URL`.

### Esito atteso
- **PASS**: `.env` minimale ma completo e non ambiguo.  
- **FAIL**: variabili equivalenti o in conflitto.

### Stato attuale dell’audit
**DA CONFERMARE IN ESECUZIONE REALE**.:contentReference[oaicite:61]{index=61}:contentReference[oaicite:62]{index=62}

---

## D3. Verificare che la terminologia usata nei documenti tecnici sia uniforme

### Perché è importante
La qualità della manutenzione crolla in silenzio quando la documentazione chiama la stessa cosa con tre nomi diversi. Classico gremlin documentale.

### Cosa validare
- “embed” vs “embeddings”;
- “test end-to-end” vs “test tecnico collection HTML”;
- “pipeline completa” vs “pipeline operativa + chat”;
- naming script e cartelle.

### Esito atteso
- **PASS**: glossario stabile.  
- **FAIL**: lo stesso componente viene descritto in modi incompatibili.

### Stato attuale dell’audit
**PARZIALMENTE FAIL** per endpoint e copertura test.:contentReference[oaicite:63]{index=63}:contentReference[oaicite:64]{index=64}:contentReference[oaicite:65]{index=65}

---

## 5. Matrice sintetica di esito

| ID | Area | Severità | Stato audit | Nota breve |
|---|---|---:|---|---|
| A1 | Chunk overlap indexer | Bloccante | FAIL probabile | Overlap dichiarato ma logicamente nullo nell’HTML indexer |
| A2 | Filtro anno documenti | Bloccante | FAIL probabile | Regex anno da URL/file sospetta |
| A3 | Endpoint embeddings Ollama | Bloccante | FAIL documentale / rischio runtime | `/api/embed` vs `/api/embeddings` non allineati |
| B1 | Copertura test pipeline finale | Alta | FAIL parziale | `test_rag.py` sembra focalizzato sulla collection HTML |
| B2 | Metrica cosine | Alta | Da confermare | Va verificata su entrambe le collection buildate |
| B3 | Requirements completi | Alta | FAIL probabile | Chat richiede `gradio`, non allineato nelle appendici principali |
| C1 | Patch contatti scraper | Media | Da confermare | Struttura buona, serve prova sul corpus generato |
| C2 | Documenti brevi | Media | Da confermare | La guida parla di patch, serve riscontro sui dati |
| C3 | Policy SOLO FONTI | Media | Da confermare | Necessari test funzionali su query coperte e scoperte |
| C4 | Runbook/orchestrazione | Media | Medio rischio | Pipeline corrente coerente, ma esistono fonti legacy |
| D1 | Fonti obsolete | Bassa | FAIL lieve | Presente documentazione non più allineata |
| D2 | Igiene `.env` | Bassa | Da confermare | Possibili override ambigui |
| D3 | Uniformità terminologica | Bassa | Parziale FAIL | Test e endpoint descritti in modo non sempre uniforme |

---

## 6. Procedura di validazione consigliata, senza modificare il codice

Questa sezione descrive **come validare**, non come correggere.

### Fase 1 — Integrità ambiente
- Confermare Python, venv, `pip install -r requirements.txt`, Ollama attivo e modelli presenti. La guida operativa richiede Python, venv, `nomic-embed-text` e `mistral`.:contentReference[oaicite:66]{index=66}
- Confermare che la chat abbia tutte le dipendenze richieste dal proprio header (`gradio`, `requests`, `chromadb`).:contentReference[oaicite:67]{index=67}

### Fase 2 — Integrità configurazione
- Verificare `.env` effettivo e confronto con template delle guide correnti.:contentReference[oaicite:68]{index=68}:contentReference[oaicite:69]{index=69}
- Confermare endpoint embeddings usato realmente dalla chat e dagli indexer.:contentReference[oaicite:70]{index=70}:contentReference[oaicite:71]{index=71}

### Fase 3 — Integrità corpus
- Eseguire scraping HTML e verificare che pagine sensibili (contatti, segreteria, orari, modulistica) compaiano nel JSONL prodotto.  
- Eseguire document crawler in dry-run e verificare classificazione anno, tipo file, count documenti. Il downloader ha funzioni dedicate a HEAD/analisi e policy anno.:contentReference[oaicite:72]{index=72}:contentReference[oaicite:73]{index=73}

### Fase 4 — Integrità indicizzazione
- Verificare metadata cosine su entrambe le collection.  
- Campionare chunk consecutivi per confermare overlap reale.  
- Verificare che documenti brevi risultino presenti.

### Fase 5 — Integrità retrieval
- Eseguire `test_rag.py` su query HTML note (“contatti segreteria”, “telefono segreteria”).:contentReference[oaicite:74]{index=74}
- Eseguire query note che dovrebbero essere presenti solo nei documenti, per misurare il gap tra test tecnico e chat finale.

### Fase 6 — Integrità risposta finale
- In chat, verificare modalità veloce e completa. La guida distingue esplicitamente le due modalità e impone fonti obbligatorie/fallback A-B-C.:contentReference[oaicite:75]{index=75}
- Validare tre classi di query:
  1. coperte da HTML;  
  2. coperte solo da DOCS;  
  3. non coperte → deve emergere NO-ANSWER istituzionale.:contentReference[oaicite:76]{index=76}

---

## 7. Domande guida per audit o passaggio consegne

Usa queste domande in un’altra chat o in una revisione tecnica:

1. Il sistema indicizza davvero due collection separate e le fonde in retrieval, oppure il test certifica solo HTML?:contentReference[oaicite:77]{index=77}:contentReference[oaicite:78]{index=78}
2. L’overlap chunk è operativo o solo dichiarato in documentazione?:contentReference[oaicite:79]{index=79}:contentReference[oaicite:80]{index=80}
3. Il filtro anno documenti riconosce davvero l’anno in URL e filename?:contentReference[oaicite:81]{index=81}
4. Tutti i componenti usano davvero lo stesso endpoint embeddings?:contentReference[oaicite:82]{index=82}:contentReference[oaicite:83]{index=83}
5. Il setup documentato installa davvero anche la chat Gradio?:contentReference[oaicite:84]{index=84}:contentReference[oaicite:85]{index=85}
6. Le risposte rispettano davvero la policy “SOLO FONTI” nei casi ambigui o scoperti?:contentReference[oaicite:86]{index=86}
7. Le guide obsolete sono state separate dalle guide correnti?:contentReference[oaicite:87]{index=87}:contentReference[oaicite:88]{index=88}

---

## 8. Verdetto finale dell’audit

### Esito complessivo
**AMBER / GIALLO SCURO**.

### Traduzione pratica
Il progetto è **credibile e ben impostato**, ma **non ancora validato in modo pieno** per produzione o consegna tecnica “senza riserve”.

### Motivi principali
- L’architettura è corretta e moderna per un RAG locale scolastico: dual collection, cosine, Ollama, fallback, prompt vincolato alle fonti.:contentReference[oaicite:89]{index=89}
- Esistono però segnali concreti di debito tecnico/di coerenza che incidono proprio sulle fondamenta del retrieval e della riproducibilità: overlap, filtro anno, endpoint embeddings, copertura test, requirements chat.:contentReference[oaicite:90]{index=90}:contentReference[oaicite:91]{index=91}:contentReference[oaicite:92]{index=92}:contentReference[oaicite:93]{index=93}

### Formula sintetica da riusare in altre chat
> Il RAG FreudKB ha una buona architettura di base, ma la validazione tecnica non è chiusa: ci sono almeno tre criticità ad alta priorità da verificare subito (chunk overlap, filtro anno documenti, coerenza endpoint embeddings), più una copertura test incompleta rispetto alla pipeline finale dual-collection e una divergenza documentale sulle dipendenze della chat.

---

## 9. Fonti usate per questo audit

### Fonti correnti principali
- `FreudKB_RAG_Guida_Completa.docx` — pipeline, `.env`, retrieval, runbook, prompt, requirements.:contentReference[oaicite:94]{index=94}:contentReference[oaicite:95]{index=95}:contentReference[oaicite:96]{index=96}
- `FreudKB_RAG_Guida_PCTO_Kevin.docx` — runbook operativo, troubleshooting, test tecnico, endpoint embeddings, patch contatti.:contentReference[oaicite:97]{index=97}:contentReference[oaicite:98]{index=98}:contentReference[oaicite:99]{index=99}
- `03_index_freud.py` — logica chunking HTML e parametri chunk.:contentReference[oaicite:100]{index=100}
- `02_download_documents.py` — stima anno da URL/file e policy di filtro documenti.:contentReference[oaicite:101]{index=101}
- `chat_ragNew.py` — dipendenze chat, endpoint embeddings, collection HTML+DOCS, parametri runtime.:contentReference[oaicite:102]{index=102}:contentReference[oaicite:103]{index=103}
- `test_rag.py` — copertura del test tecnico, verifica embedding 768, distances e top-5.:contentReference[oaicite:104]{index=104}

### Fonte legacy da trattare come non canonica
- `Guida_FreudKB_Completa_v2.docx` — contiene riferimenti a script non allineati alla pipeline corrente (`01_enrich_kb.py`, `02_index_chroma.py`).:contentReference[oaicite:105]{index=105}

---

## 10. Allegato pronto da copiare in un’altra chat

Puoi incollare questo prompt operativo:

> Analizza questo progetto RAG locale FreudKB senza modificare il codice. Considera come priorità di audit: 1) chunk overlap reale negli indexer, 2) correttezza filtro anno in `02_download_documents.py`, 3) coerenza endpoint embeddings Ollama tra `.env`, test e `chat_ragNew.py`, 4) copertura reale del test rispetto alla pipeline dual-collection HTML+DOCS, 5) sufficienza del `requirements.txt` per avviare anche la chat Gradio, 6) rispetto della policy “SOLO FONTI”. Produci una diagnosi ordinata per severità con evidenze, stato PASS/FAIL/DA CONFERMARE e rischio operativo.

---

Fine documento.