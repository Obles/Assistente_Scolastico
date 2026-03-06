# scripts/04_run_pipeline.py
# Esegue la pipeline Freud-KB in sequenza
# Permette di scegliere lo step da cui partire:
#    python 04_run_pipeline.py 02
# (parte da 02 e prosegue fino a 03b)

import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
LOGFILE = ROOT / "pipeline.log"

# Mappa ordinata degli step
# Chiave = “codice”, Valore = script.py
STEPS = {
    "00": "00_clean_freud_kb.py",
    "01": "01_scrape_freud.py",
    "02": "02_download_documents.py",
    "03": "03_index_freud.py",
    "03b": "03b_index_documents.py",
}

def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOGFILE.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def run_step(script_name):
    script_path = SCRIPTS / script_name

    log(f"START  {script_name}")
    print(f"\n=== ESEGUO {script_name} ===")

    result = subprocess.run([sys.executable, str(script_path)])

    log(f"END    {script_name} (code {result.returncode})")

    if result.returncode != 0:
        print(f"[ERRORE] {script_name} fallito con codice {result.returncode}")
        sys.exit(result.returncode)

    print(f"✔ Completato {script_name}")

def main():
    print("========================================")
    print(" PIPELINE FREUD-KB — AVVIO")
    print(" Log: pipeline.log")
    print("========================================")

    # Step di partenza (default = 00)
    start_arg = sys.argv[1] if len(sys.argv) > 1 else "00"
    start_arg = start_arg.lower()

    # Validazione input
    if start_arg not in STEPS:
        print(f"[ERRORE] Step di partenza non valido: {start_arg}")
        print(f"Valori ammessi: {', '.join(STEPS.keys())}")
        sys.exit(1)

    # Ottieni lista steps ordinata dal codice di partenza
    keys = list(STEPS.keys())
    start_index = keys.index(start_arg)
    steps_to_run = keys[start_index:]

    print(f">> Avvio da step: {start_arg}")
    log(f"PIPELINE START (from {start_arg})")

    # Esecuzione sequenziale
    for key in steps_to_run:
        run_step(STEPS[key])

    print("\n========================================")
    print("✔ PIPELINE COMPLETATA SENZA ERRORI")
    print("========================================")
    log("PIPELINE COMPLETATA (OK)")

if __name__ == "__main__":
    main()