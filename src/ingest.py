"""
ingest.py
HOA RAG Chatbot - Document Ingestion Script

Ingests HOA content from two sources:
  1. S3 bucket  — all .pdf and .docx files are downloaded and chunked automatically.
                  Add a new doc to the bucket; the next ingest picks it up with no
                  code changes needed.
  2. Web pages  — comma-separated URLs in SCRAPE_URLS are fetched, parsed, and
                  chunked by heading. Useful for fee schedules and other public pages.

Requires S3_BUCKET in .env or environment. Uses EC2 IAM role / local AWS
credentials — no hardcoded keys.

Usage:
  python src/ingest.py
"""

import os
import re
import sys
import shutil
import tempfile
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
SCRAPE_URLS = [u.strip() for u in os.getenv("SCRAPE_URLS", "").split(",") if u.strip()]
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hoa_documents"

if not S3_BUCKET:
    sys.exit(
        "Error: S3_BUCKET is not set.\n"
        "Add  S3_BUCKET=your-bucket-name  to your .env file or environment."
    )

EMBEDDING_FN = embedding_functions.DefaultEmbeddingFunction()


# ── S3 ────────────────────────────────────────────────────────────────────────
def list_docs_in_s3(bucket: str) -> list[str]:
    """Return all .pdf and .docx object keys in the bucket."""
    import boto3
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".pdf", ".docx")):
                keys.append(key)
    return keys


def download_docs_from_s3(bucket: str, keys: list[str], tmp_dir: str):
    import boto3
    s3 = boto3.client("s3")
    print(f"Downloading {len(keys)} document(s) from s3://{bucket}/")
    for key in keys:
        dest = os.path.join(tmp_dir, os.path.basename(key))
        print(f"  ↓ {key}")
        s3.download_file(bucket, key, dest)
    print(f"  ✅ All documents downloaded.\n")


# ── PDF Reader ────────────────────────────────────────────────────────────────
def read_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() for page in reader.pages)


# ── Word Reader ───────────────────────────────────────────────────────────────
def read_docx(filepath: str) -> list[dict]:
    """Returns list of {text, is_bold} dicts, one per non-empty paragraph."""
    doc = Document(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            is_bold = any(run.bold for run in para.runs if run.text.strip())
            paragraphs.append({"text": para.text.strip(), "is_bold": is_bold})
    return paragraphs


# ── Chunkers ──────────────────────────────────────────────────────────────────
def chunk_bylaws(text: str) -> list[dict]:
    """Splits by ARTICLE and Section; preserves structure for citations."""
    chunks = []
    current_article = "Preamble"
    current_section = ""
    current_text = []

    def save_chunk():
        body = " ".join(current_text).strip()
        if body:
            chunks.append({
                "text": body,
                "source": "Bylaws",
                "article": current_article,
                "section": current_section,
                "citation": f"Bylaws, {current_article}{', ' + current_section if current_section else ''}"
            })

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        article_match = re.match(r"^(ARTICLE\s+[IVXLCDM]+\.?\s*.+)$", line, re.IGNORECASE)
        section_match = re.match(r"^(Section\s+\d+\.?\s*.*)$", line, re.IGNORECASE)
        if article_match:
            save_chunk()
            current_article = article_match.group(1).strip()
            current_section = ""
            current_text = []
        elif section_match:
            save_chunk()
            current_section = section_match.group(1).strip()
            current_text = []
        else:
            current_text.append(line)

    save_chunk()
    return chunks


def chunk_by_numbered_sections(text: str, source_name: str) -> list[dict]:
    """Splits on numbered headers like '1. GENERAL RULES'."""
    chunks = []
    current_section = "General"
    current_text = []

    def save_chunk():
        body = " ".join(current_text).strip()
        if body:
            chunks.append({
                "text": body,
                "source": source_name,
                "article": "",
                "section": current_section,
                "citation": f"{source_name}, {current_section}"
            })

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        numbered_match = re.match(r"^(\d+)\.\s+([A-Z][^\n]{3,})$", line)
        if numbered_match:
            save_chunk()
            current_section = f"Section {numbered_match.group(1)}: {numbered_match.group(2).strip()}"
            current_text = []
        else:
            current_text.append(line)

    save_chunk()
    return chunks


def chunk_docx_by_bold_headers(paragraphs: list[dict], source_name: str) -> list[dict]:
    """Splits Word docs on bold header lines."""
    chunks = []
    current_section = "General"
    current_text = []

    def save_chunk():
        body = " ".join(current_text).strip()
        if body:
            chunks.append({
                "text": body,
                "source": source_name,
                "article": "",
                "section": current_section,
                "citation": f"{source_name}, {current_section}"
            })

    for para in paragraphs:
        if para["is_bold"] and len(para["text"]) < 100:
            save_chunk()
            current_section = para["text"]
            current_text = []
        else:
            current_text.append(para["text"])

    save_chunk()
    return chunks


def chunk_by_caps_headers(text: str, source_name: str) -> list[dict]:
    """
    Splits PDFs that use ALL-CAPS lines as section headers
    (e.g. history documents, general info sheets).
    Ignores page headers/footers (very short all-caps lines < 5 words).
    """
    chunks = []
    current_section = "General"
    current_text = []

    def save_chunk():
        body = " ".join(current_text).strip()
        if body and len(body) > 60:
            chunks.append({
                "text": body,
                "source": source_name,
                "article": "",
                "section": current_section,
                "citation": f"{source_name}, {current_section}"
            })

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        word_count = len(stripped.split())
        is_caps_header = (
            stripped.isupper()
            and 2 <= word_count <= 12
            and not re.search(r'\d{4}', stripped)  # skip lines that are just years/dates
        )
        if is_caps_header:
            save_chunk()
            current_section = stripped.title()
            current_text = []
        else:
            current_text.append(stripped)

    save_chunk()
    return chunks


def chunk_by_paragraphs(text: str, source_name: str, chunk_size: int = 500) -> list[dict]:
    """Fallback chunker: splits by paragraph with overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        current_chunk.append(para)
        current_len += len(para)
        if current_len >= chunk_size:
            body = " ".join(current_chunk).strip()
            chunks.append({
                "text": body,
                "source": source_name,
                "article": "",
                "section": body[:60] + "...",
                "citation": source_name
            })
            current_chunk = [current_chunk[-1]]
            current_len = len(current_chunk[0])

    if current_chunk:
        body = " ".join(current_chunk).strip()
        if body:
            chunks.append({
                "text": body,
                "source": source_name,
                "article": "",
                "section": body[:60] + "...",
                "citation": source_name
            })

    return chunks


# ── Web Scraper ───────────────────────────────────────────────────────────────
def _split_by_dept_headers(text: str, page_title: str, url: str, parent_heading: str) -> list[dict]:
    """
    Within a block of table text, split on ALL-CAPS department header lines
    (e.g. 'NAUTICAL DEPARTMENT', 'TENNIS DEPARTMENT') so each department gets
    its own chunk rather than everything landing in one blob.
    """
    chunks = []
    current_section = parent_heading
    current_lines: list[str] = []

    def flush():
        body = "\n".join(l for l in current_lines if l.strip())
        if body and len(body) > 40:  # skip breadcrumb-length junk
            chunks.append({
                "text": body,
                "source": page_title,
                "article": "",
                "section": current_section,
                "citation": f"{page_title} — {current_section} ({url})"
            })

    for line in text.splitlines():
        stripped = line.strip()
        # Detect ALL-CAPS section headers (no pipe, not purely punctuation/numbers)
        is_dept_header = (
            stripped
            and re.match(r'^[A-Z][A-Z\s,\.]+$', stripped)
            and "|" not in stripped
            and len(stripped) > 4
        )
        if is_dept_header:
            flush()
            current_section = stripped.title()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return chunks


def scrape_url(url: str) -> list[dict]:
    """
    Fetches a public URL and chunks its content by heading (h1–h4).
    Fee tables are further split by department header lines so each
    department gets its own retrievable chunk.
    """
    import requests
    from bs4 import BeautifulSoup

    print(f"  Fetching {url}")
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    page_title = soup.title.get_text(strip=True) if soup.title else url

    chunks: list[dict] = []
    current_heading = "General"
    current_lines: list[str] = []
    pending_tables: list[str] = []  # table text collected under current heading

    def flush():
        # Flush any accumulated paragraph text
        body = " ".join(current_lines).strip()
        if body and len(body) > 40:
            chunks.append({
                "text": body,
                "source": page_title,
                "article": "",
                "section": current_heading,
                "citation": f"{page_title} — {current_heading} ({url})"
            })
        # Flush tables, split internally by dept headers
        for tbl_text in pending_tables:
            sub = _split_by_dept_headers(tbl_text, page_title, url, current_heading)
            chunks.extend(sub)

    def table_to_text(table_tag) -> str:
        rows = []
        for tr in table_tag.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append("  |  ".join(cells))
        return "\n".join(rows)

    visited_tables: set[int] = set()

    body_tag = soup.find("body") or soup
    for element in body_tag.descendants:
        if not hasattr(element, "name"):
            continue

        if element.name in ("h1", "h2", "h3", "h4"):
            flush()
            current_heading = element.get_text(" ", strip=True)
            current_lines = []
            pending_tables.clear()

        elif element.name == "table":
            if id(element) not in visited_tables:
                visited_tables.add(id(element))
                pending_tables.append(table_to_text(element))

        elif element.name == "p":
            text = element.get_text(" ", strip=True)
            if text:
                current_lines.append(text)

        elif element.name == "li":
            text = element.get_text(" ", strip=True)
            if text:
                current_lines.append(f"• {text}")

    flush()
    return chunks


# ── Per-file dispatch ─────────────────────────────────────────────────────────
def _source_name(filepath: str) -> str:
    """Derive a readable source name from a filename."""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return stem.replace("_", " ").replace("-", " ").title()


def chunk_file(filepath: str) -> list[dict]:
    """
    Routes each file to the right chunker based on filename.
    Known files get their original strategy; new files get a sensible default
    (numbered sections for PDFs, bold headers for DOCX).
    """
    filename = os.path.basename(filepath).lower()
    ext = os.path.splitext(filename)[1]
    name = _source_name(filepath)

    if filename == "bylaws.pdf":
        return chunk_bylaws(read_pdf(filepath))
    elif "covenant" in filename and ext == ".pdf":
        return chunk_by_paragraphs(read_pdf(filepath), name)
    elif ext == ".pdf":
        text = read_pdf(filepath)
        chunks = chunk_by_numbered_sections(text, name)
        if len(chunks) < 5:
            # Numbered sections didn't split well — try ALL-CAPS headers
            caps_chunks = chunk_by_caps_headers(text, name)
            if len(caps_chunks) > len(chunks):
                chunks = caps_chunks
        if len(chunks) < 5:
            # Last resort: paragraph-based splitting
            chunks = chunk_by_paragraphs(text, name)
        return chunks
    elif ext == ".docx":
        return chunk_docx_by_bold_headers(read_docx(filepath), name)
    else:
        print(f"  ⚠ Skipping unsupported file type: {filename}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────
def ingest_all():
    print(f"Starting HOA document ingestion from s3://{S3_BUCKET}/\n")

    keys = list_docs_in_s3(S3_BUCKET)
    if not keys:
        sys.exit(f"No .pdf or .docx files found in s3://{S3_BUCKET}/")

    tmp_dir = tempfile.mkdtemp()
    try:
        download_docs_from_s3(S3_BUCKET, keys, tmp_dir)

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Cleared existing ChromaDB collection.")
        except Exception:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=EMBEDDING_FN
        )

        all_chunks = []

        # S3 documents
        for key in keys:
            filepath = os.path.join(tmp_dir, os.path.basename(key))
            print(f"Processing {os.path.basename(key)}...")
            chunks = chunk_file(filepath)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks from {_source_name(filepath)}")

        # Web pages
        if SCRAPE_URLS:
            print(f"\nScraping {len(SCRAPE_URLS)} web page(s)...")
            for url in SCRAPE_URLS:
                try:
                    chunks = scrape_url(url)
                    all_chunks.extend(chunks)
                    print(f"  → {len(chunks)} chunks from {url}")
                except Exception as exc:
                    print(f"  ⚠ Failed to scrape {url}: {exc}")

        print(f"\nStoring {len(all_chunks)} total chunks in ChromaDB...")
        for i, chunk in enumerate(all_chunks):
            collection.add(
                ids=[f"chunk_{i}"],
                documents=[chunk["text"]],
                metadatas=[{
                    "source": chunk["source"],
                    "article": chunk.get("article", ""),
                    "section": chunk.get("section", ""),
                    "citation": chunk.get("citation", chunk["source"])
                }]
            )

        print(f"\n✅ Ingestion complete!")
        print(f"   {len(all_chunks)} chunks stored in '{CHROMA_DIR}/'")
        print(f"   {len(keys)} S3 document(s) + {len(SCRAPE_URLS)} web page(s) processed")
        print(f"   Ready to query with query.py")

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    ingest_all()
