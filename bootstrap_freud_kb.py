import os, textwrap
from pathlib import Path

root = Path('freud-kb')
(data_dir := root / 'data').mkdir(parents=True, exist_ok=True)
(build_dir := root / 'build' / 'pages').mkdir(parents=True, exist_ok=True)
(scripts_dir := root / 'scripts').mkdir(parents=True, exist_ok=True)
(root / 'build' / 'chroma_freud').mkdir(parents=True, exist_ok=True)

readme = textwrap.dedent('''# Freud KB – Zero‑Cost RAG Pipeline
[... omesso per brevità: usa il README che ti ho già messo nel progetto generato ...]
''')

script_01 = """# scripts/00_clean_freud_kb.py
[... inserisci qui il contenuto completo dello script Enrichment ...]
"""

script_02 = """# scripts/01_scrape_freud.py
[... inserisci qui il contenuto completo dello script Index Chroma ...]
"""

script_03 = """# scripts/02_download_documents.py
[... inserisci qui il contenuto completo dello script Index Chroma ...]
"""
script_04 = """# scripts/03_index_freud.py
[... inserisci qui il contenuto completo dello script Index Chroma ...]
"""
script_05 = """# scripts/03b_index_documents.py
[... inserisci qui il contenuto completo dello script Index Chroma ...]
"""
script_06 = """# scripts/common_config.py
[... inserisci qui il contenuto completo dello script Index Chroma ...]
"""
script_07 = """# scripts/04_run_pipeline.py
[... inserisci qui il contenuto completo dello script Index Chroma ...]
"""
script_08 = """# scripts/chat_ragNew.py
[... inserisci qui il contenuto completo dello script Index Chroma ...]
"""

(root / 'README.md').write_text(readme, encoding='utf-8')
(scripts_dir / '00_clean_freud_kb.py').write_text(script_01, encoding='utf-8')
(scripts_dir / '01_scrape_freud.py').write_text(script_02, encoding='utf-8')
(scripts_dir / '02_download_documents.py').write_text(script_03, encoding='utf-8')
(scripts_dir / '03_index_freud.py').write_text(script_04, encoding='utf-8')
(scripts_dir / '03b_index_documents.py').write_text(script_05, encoding='utf-8')
(scripts_dir / 'common_config.py').write_text(script_06, encoding='utf-8')
(scripts_dir / '04_run_pipeline.py').write_text(script_07, encoding='utf-8')
(scripts_dir / 'chat_ragNew.py').write_text(script_08, encoding='utf-8')
print("Project skeleton created at", root.resolve())