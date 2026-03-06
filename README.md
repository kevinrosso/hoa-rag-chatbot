# Ulmstead Club HOA Assistant

A RAG-based chatbot for HOA members to ask questions about policies, bylaws, and guidelines.

## Project Structure
- `app.py` - Streamlit chat interface
- `src/ingest.py` - Document ingestion and ChromaDB setup
- `src/query.py` - Search and Claude-powered answer logic
- `docs/` - HOA documents (not committed, kept local/S3)
- `chroma_db/` - Generated vector database (not committed)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# 1. Add HOA documents to docs/ folder
# 2. Run ingestion once
python src/ingest.py

# 3. Launch chat interface
streamlit run app.py
```

## Environment Variables
Create a `.env` file with:
```
ANTHROPIC_API_KEY=your-api-key-here
```
