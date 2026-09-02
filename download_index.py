"""
download_index.py — Fetches a pre-built data/ directory (chunks.jsonl,
bm25_index.pkl, chroma_db/, corpus_metadata.json) from a release asset so
the Streamlit app loads without visitors waiting on a full re-ingest.

Fill in RELEASE_URL after you've run src/ingest.py once locally and
uploaded the resulting data/ directory (zipped) as a GitHub release asset
or other stable host. If data/chroma_db/ is under ~100MB, it's simpler to
just commit it directly and skip this script entirely.
"""

import shutil
import zipfile
from pathlib import Path

import requests

RELEASE_URL = "https://github.com/<your-username>/hep-rag/releases/download/v1.0/data.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"


def download_index():
    if (DATA_DIR / "chroma_db").exists():
        print("Index already present, skipping download.")
        return

    print(f"Downloading pre-built index from {RELEASE_URL} ...")
    zip_path = Path("/tmp/hep_rag_data.zip")
    r = requests.get(RELEASE_URL, timeout=120)
    r.raise_for_status()
    zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR.parent)

    zip_path.unlink()
    print("Index downloaded and extracted.")


if __name__ == "__main__":
    download_index()
