"""
query.py
HOA RAG Chatbot - Query and Answer Script

Takes a user question, searches ChromaDB for relevant HOA document sections,
and uses Claude to generate a cited answer.

Usage:
  Interactive mode:  python src/query.py
  Single question:   python src/query.py "Can a non-member use the boat dock?"
"""

import os
import sys
import chromadb
from chromadb.utils import embedding_functions
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hoa_documents"
TOP_K = 8  # Number of most relevant chunks to retrieve per question

EMBEDDING_FN = embedding_functions.DefaultEmbeddingFunction()

# ── Claude System Prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant for Ulmstead Club, Inc., an Arnold, MD community.
It is a community-run, non-profit organization that operates like a typical HOA.
Your job is to answer questions from community members about community policies, bylaws, and guidelines.
Keep answers clear and concise.

You will be given relevant excerpts from the official UCI documents with each question.
You may also be given recent conversation history for context.

When answering:
1. Base your answer ONLY on the provided document excerpts — they always take precedence over
   anything discussed earlier in the conversation
2. You may use conversation history to understand what the member is referring to
   (e.g. follow-up questions like "how much is that?" or "what about renters?")
3. Always cite the specific document, article, and section when available
   Example: "Per Bylaws Article I, Section 1..." or "Per Nautical Guidelines, Section 2..."
4. If multiple documents are relevant, reference all of them
5. If the excerpts do not contain a clear answer, say so honestly:
   "There is no specific article or policy that directly addresses this. Based on the
   overall intent of the bylaws/policies, this would likely be interpreted as [your interpretation]."
6. Keep answers clear and concise — members should be able to understand them easily
7. Never make up rules or policies that are not in the provided excerpts
8. If a question is clearly outside the scope of HOA documents, politely say so"""

HISTORY_LIMIT = 6  # number of recent messages (user + assistant) to include


# ── Load ChromaDB ─────────────────────────────────────────────────────────────
def load_collection():
    """Load the ChromaDB collection. Raises clear error if ingest hasn't been run."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_DIR}/'. "
            "Please run 'python src/ingest.py' first."
        )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FN
    )
    return collection


def get_sources(collection) -> list[str]:
    """Return sorted unique source names from the collection."""
    results = collection.get(include=["metadatas"])
    sources = sorted({m.get("source", "") for m in results["metadatas"] if m.get("source")})
    return sources


# ── Search ────────────────────────────────────────────────────────────────────
def search_documents(collection, question: str) -> list[dict]:
    """
    Search ChromaDB for the most relevant document chunks.
    Returns list of {text, citation, source, article, section} dicts.
    """
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "citation": results["metadatas"][0][i].get("citation", "Unknown"),
            "source": results["metadatas"][0][i].get("source", ""),
            "article": results["metadatas"][0][i].get("article", ""),
            "section": results["metadatas"][0][i].get("section", ""),
            "distance": results["distances"][0][i] if "distances" in results else None
        })

    return chunks


# ── Build Context ─────────────────────────────────────────────────────────────
def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for Claude."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} — {chunk['citation']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


# ── Answer ────────────────────────────────────────────────────────────────────
def answer_question(question: str, collection, history=None) -> str:
    """
    Full RAG pipeline with optional conversation history.
    history: list of {"role": "user"|"assistant", "content": "..."} dicts,
             most recent last, capped to HISTORY_LIMIT before use.
    """
    chunks = search_documents(collection, question)

    if not chunks:
        return "I was unable to find any relevant information in the HOA documents for your question."

    context = build_context(chunks)

    current_message = f"""Here are the relevant excerpts from the HOA documents:

{context}

---

Member question: {question}

Please answer the question based on the excerpts above, citing the specific document and section."""

    # Build messages: prior turns (plain text) + current turn (with docs injected)
    messages = []
    if history:
        for msg in history[-HISTORY_LIMIT:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": current_message})

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    return response.content[0].text


# ── Interactive CLI ───────────────────────────────────────────────────────────
def run_interactive(collection):
    """Simple interactive command-line chat loop."""
    print("\n🏘️  Ulmstead Community Assistant")
    print("=" * 40)
    print("Ask any question about UCI policies, bylaws, or guidelines.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print("\nSearching documents...\n")
        answer = answer_question(question, collection)
        print(f"Answer:\n{answer}")
        print("\n" + "-" * 40 + "\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        collection = load_collection()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Single question mode: python src/query.py "your question here"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nQuestion: {question}\n")
        answer = answer_question(question, collection)
        print(f"Answer:\n{answer}\n")

    # Interactive mode: python src/query.py
    else:
        run_interactive(collection)