"""
ingest.py — Corpus acquisition, parsing, chunking, and dual index construction.

Pipeline:
    arXiv API query -> download PDFs -> section-aware parse (PyMuPDF)
    -> hierarchical chunking -> SPECTER2 dense index (ChromaDB)
    -> BM25 sparse index (pickled)

Usage:
    python src/ingest.py --query "cat:hep-ex AND all:missing transverse energy dark matter" \
                          --max-results 80 --out-dir data/

Run this once. The resulting data/ directory (corpus_metadata.json,
chunks.jsonl, bm25_index.pkl, chroma_db/) is what the Streamlit app and
eval harness read from — it is NOT regenerated on every app launch.
"""

import argparse
import json
import pickle
import re
import time
import urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Common HEP paper section headers used as chunk-boundary delimiters.
# Matched case-insensitively, at the start of a line, optionally numbered
# ("3. Event Selection", "III. RESULTS", "Event Selection").
SECTION_HEADERS = [
    "abstract", "introduction", "data", "data and simulation", "event selection",
    "background", "background estimation", "signal region", "control region",
    "results", "systematic uncertainties", "systematics", "interpretation",
    "conclusion", "conclusions", "summary", "acknowledgments", "references",
]
SECTION_RE = re.compile(
    r"^\s*(?:[IVXLC]+\.|\d+\.?)?\s*(" + "|".join(re.escape(s) for s in SECTION_HEADERS) + r")\s*$",
    re.IGNORECASE,
)
EQ_REF_RE = re.compile(r"\bEq(?:uation)?s?\.?\s*\(?(\d+)\)?", re.IGNORECASE)
FIG_REF_RE = re.compile(r"\bFig(?:ure)?s?\.?\s*(\d+)", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    section: str
    chunk_index: int
    text: str
    token_count: int
    equation_refs: list
    figure_refs: list


def fetch_arxiv_metadata(search_query: str, max_results: int = 80) -> list[dict]:
    """Query the arXiv API and return a list of paper metadata dicts."""
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id_full = entry.find("atom:id", ATOM_NS).text.strip()
        arxiv_id = arxiv_id_full.rsplit("/", 1)[-1]
        # strip version suffix (v1, v2, ...) for a stable id, keep it for the PDF url
        title = " ".join(entry.find("atom:title", ATOM_NS).text.split())
        abstract = " ".join(entry.find("atom:summary", ATOM_NS).text.split())
        authors = [a.find("atom:name", ATOM_NS).text for a in entry.findall("atom:author", ATOM_NS)]
        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]
        if pdf_url is None:
            pdf_url = arxiv_id_full.replace("abs", "pdf")

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "pdf_url": pdf_url,
        })
    return papers


def download_pdfs(papers: list[dict], out_dir: Path) -> list[dict]:
    """Download PDFs, skip + log 404s/failures. Returns papers that succeeded."""
    pdf_dir = out_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    failed = []
    for p in tqdm(papers, desc="Downloading PDFs"):
        dest = pdf_dir / f"{p['arxiv_id'].replace('/', '_')}.pdf"
        try:
            r = requests.get(p["pdf_url"], timeout=60)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                dest.write_bytes(r.content)
                p["local_pdf"] = str(dest)
                ok.append(p)
            else:
                failed.append((p["arxiv_id"], r.status_code))
        except requests.RequestException as e:
            failed.append((p["arxiv_id"], str(e)))
        time.sleep(0.5)  # be polite to arXiv

    print(f"Downloaded {len(ok)}/{len(papers)} PDFs. Failed: {failed}")
    return ok


def extract_sections(pdf_path: str) -> list[tuple[str, str]]:
    """
    Parse a PDF into (section_name, section_text) pairs using PyMuPDF.
    Falls back to a single 'Full Text' section if no headers are detected.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    lines = full_text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_name = "Preamble"
    current_lines: list[str] = []

    for line in lines:
        m = SECTION_RE.match(line.strip())
        if m:
            sections.append((current_name, current_lines))
            current_name = m.group(1).title()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_name, current_lines))

    result = [(name, "\n".join(txt).strip()) for name, txt in sections if "\n".join(txt).strip()]
    if len(result) <= 1:
        return [("Full Text", full_text)]
    return result


def chunk_section(text: str, chunk_tokens: int = 256, overlap_tokens: int = 64) -> list[str]:
    """
    Level-2 dense chunking: ~256-token windows with 64-token overlap.
    Uses a simple whitespace-token approximation (good enough for chunk
    boundaries; the embedding model does its own subword tokenization).
    """
    words = text.split()
    if not words:
        return []
    step = max(chunk_tokens - overlap_tokens, 1)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_tokens]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_tokens >= len(words):
            break
    return chunks


def build_chunks(papers: list[dict]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for p in tqdm(papers, desc="Parsing + chunking"):
        try:
            sections = extract_sections(p["local_pdf"])
        except Exception as e:
            print(f"  ! failed to parse {p['arxiv_id']}: {e}")
            continue

        for section_name, section_text in sections:
            pieces = chunk_section(section_text)
            for i, piece in enumerate(pieces):
                eq_refs = sorted(set(f"Eq. {n}" for n in EQ_REF_RE.findall(piece)))
                fig_refs = sorted(set(f"Fig. {n}" for n in FIG_REF_RE.findall(piece)))
                all_chunks.append(Chunk(
                    chunk_id=f"arxiv_{p['arxiv_id']}_s{len(all_chunks)}_c{i}",
                    paper_id=p["arxiv_id"],
                    title=p["title"],
                    section=section_name,
                    chunk_index=i,
                    text=piece,
                    token_count=len(piece.split()),
                    equation_refs=eq_refs,
                    figure_refs=fig_refs,
                ))
    return all_chunks


def build_dense_index(chunks: list[Chunk], out_dir: Path):
    """Embed all chunks with SPECTER2 and persist to a ChromaDB collection."""
    import chromadb
    from specter2_encoder import Specter2Encoder, PROXIMITY_ADAPTER

    encoder = Specter2Encoder(adapter_name=PROXIMITY_ADAPTER)

    client = chromadb.PersistentClient(path=str(out_dir / "chroma_db"))
    collection = client.get_or_create_collection(
        "hep_chunks_specter2", metadata={"hnsw:space": "cosine"}
    )

    batch = 16  # smaller batch than before: full transformer forward pass, not a lightweight sentence-transformers call
    for i in tqdm(range(0, len(chunks), batch), desc="Embedding (SPECTER2)"):
        batch_chunks = chunks[i:i + batch]
        # SPECTER2 expects "title [SEP] abstract"-style input; we use "title [SEP] chunk_text"
        texts = [f"{c.title}{encoder.sep_token}{c.text}" for c in batch_chunks]
        embeddings = encoder.encode(texts)
        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=embeddings,
            documents=[c.text for c in batch_chunks],
            metadatas=[{
                "paper_id": c.paper_id, "title": c.title, "section": c.section,
                "chunk_index": c.chunk_index,
            } for c in batch_chunks],
        )
    print(f"Dense index built: {collection.count()} vectors.")


def build_sparse_index(chunks: list[Chunk], out_dir: Path):
    from rank_bm25 import BM25Okapi

    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(out_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c.chunk_id for c in chunks],
            "tokenized_corpus": tokenized,
        }, f)
    print(f"Sparse index built: {len(chunks)} documents.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="cat:hep-ex AND all:missing transverse energy dark matter")
    ap.add_argument("--max-results", type=int, default=80)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Querying arXiv: {args.query}")
    papers = fetch_arxiv_metadata(args.query, args.max_results)
    print(f"Found {len(papers)} candidate papers.")

    ok_papers = download_pdfs(papers, out_dir)

    with open(out_dir / "corpus_metadata.json", "w") as f:
        json.dump([{k: v for k, v in p.items() if k != "local_pdf"} for p in ok_papers], f, indent=2)

    chunks = build_chunks(ok_papers)
    print(f"Built {len(chunks)} chunks from {len(ok_papers)} papers "
          f"({len(chunks) / max(len(ok_papers), 1):.1f} chunks/paper avg).")

    with open(out_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    build_dense_index(chunks, out_dir)
    build_sparse_index(chunks, out_dir)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()