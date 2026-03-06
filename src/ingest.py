"""
ingest.py
HOA RAG Chatbot - Document Ingestion Script

Reads all HOA documents, chunks them intelligently by their structure,
and stores them in a local ChromaDB vector database.

Run this once locally, and again on the server after deployment.
Usage: python src/ingest.py
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hoa_documents"

# Uses ChromaDB's built-in sentence-transformers embeddings (free, local, no API key)
EMBEDDING_FN = embedding_functions.DefaultEmbeddingFunction()


# ── PDF Reader ────────────────────────────────────────────────────────────────
def read_pdf(filepath: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ── Word Reader ───────────────────────────────────────────────────────────────
def read_docx(filepath: str) -> list[dict]:
    """
    Extract paragraphs from a Word doc, preserving bold header info.
    Returns list of {text, is_bold} dicts.
    """
    doc = Document(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            is_bold = any(run.bold for run in para.runs if run.text.strip())
            paragraphs.append({"text": para.text.strip(), "is_bold": is_bold})
    return paragraphs


# ── Bylaws Chunker ────────────────────────────────────────────────────────────
def chunk_bylaws(text: str) -> list[dict]:
    """
    Splits bylaws PDF by ARTICLE and Section.
    Each chunk gets article + section metadata for precise citations.
    """
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

        # Match ARTICLE I., ARTICLE II., etc.
        article_match = re.match(r"^(ARTICLE\s+[IVXLCDM]+\.?\s*.+)$", line, re.IGNORECASE)
        # Match Section 1., Section 2., etc.
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

    save_chunk()  # save last chunk
    return chunks


# ── Numbered Section Chunker (Nautical, Architecture, Covenants) ──────────────
def chunk_by_numbered_sections(text: str, source_name: str) -> list[dict]:
    """
    Splits documents that use numbered sections like:
    '1. GENERAL RULES' or '2. SLIP USE RESTRICTIONS'
    """
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

        # Match lines like "1. SECTION NAME" or "2. SLIP USE RESTRICTIONS –"
        numbered_match = re.match(r"^(\d+)\.\s+([A-Z][^\n]{3,})$", line)

        if numbered_match:
            save_chunk()
            current_section = f"Section {numbered_match.group(1)}: {numbered_match.group(2).strip()}"
            current_text = []
        else:
            current_text.append(line)

    save_chunk()
    return chunks


# ── Bold Header Chunker (Word docs) ───────────────────────────────────────────
def chunk_docx_by_bold_headers(paragraphs: list[dict], source_name: str) -> list[dict]:
    """
    Splits Word documents by bold headers.
    Each bold line starts a new chunk.
    """
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


# ── Paragraph Chunker (fallback) ──────────────────────────────────────────────
def chunk_by_paragraphs(text: str, source_name: str, chunk_size: int = 500) -> list[dict]:
    """
    Fallback chunker: splits by paragraph with overlap.
    Used for covenants or any doc without clear structure.
    """
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
                "section": body[:60] + "...",  # first 60 chars as section label
                "citation": source_name
            })
            # Overlap: keep last paragraph for context continuity
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


# ── Main Ingestion ────────────────────────────────────────────────────────────
def ingest_all():
    print("Starting HOA document ingestion...\n")

    # Set up ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if re-running (fresh ingest)
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

    # ── bylaws.pdf ──
    print("Processing bylaws.pdf...")
    text = read_pdf(os.path.join(DOCS_DIR, "bylaws.pdf"))
    chunks = chunk_bylaws(text)
    all_chunks.extend(chunks)
    print(f"  → {len(chunks)} chunks from Bylaws")

    # ── covenants.pdf ──
    print("Processing covenants.pdf...")
    text = read_pdf(os.path.join(DOCS_DIR, "covenants.pdf"))
    chunks = chunk_by_paragraphs(text, "Covenants")
    all_chunks.extend(chunks)
    print(f"  → {len(chunks)} chunks from Covenants")

    # ── guidelines_architecture.pdf ──
    print("Processing guidelines_architecture.pdf...")
    text = read_pdf(os.path.join(DOCS_DIR, "guidelines_architecture.pdf"))
    chunks = chunk_by_numbered_sections(text, "Architecture Guidelines")
    all_chunks.extend(chunks)
    print(f"  → {len(chunks)} chunks from Architecture Guidelines")

    # ── guidelines_nautical.pdf ──
    print("Processing guidelines_nautical.pdf...")
    text = read_pdf(os.path.join(DOCS_DIR, "guidelines_nautical.pdf"))
    chunks = chunk_by_numbered_sections(text, "Nautical Guidelines")
    all_chunks.extend(chunks)
    print(f"  → {len(chunks)} chunks from Nautical Guidelines")

    # ── guidelines_beachpark.docx ──
    print("Processing guidelines_beachpark.docx...")
    paragraphs = read_docx(os.path.join(DOCS_DIR, "guidelines_beachpark.docx"))
    chunks = chunk_docx_by_bold_headers(paragraphs, "Beach Park Guidelines")
    all_chunks.extend(chunks)
    print(f"  → {len(chunks)} chunks from Beach Park Guidelines")

    # ── policy_membership.docx ──
    print("Processing policy_membership.docx...")
    paragraphs = read_docx(os.path.join(DOCS_DIR, "policy_membership.docx"))
    chunks = chunk_docx_by_bold_headers(paragraphs, "Membership Policy")
    all_chunks.extend(chunks)
    print(f"  → {len(chunks)} chunks from Membership Policy")

    # ── Store all chunks in ChromaDB ──
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
    print(f"   Ready to query with query.py")


if __name__ == "__main__":
    ingest_all()