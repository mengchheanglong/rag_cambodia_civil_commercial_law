# AGENTS.md — RAG Cambodia Law

## Project Overview
A Retrieval-Augmented Generation (RAG) system for Cambodian Civil & Commercial Law.
The system ingests official legal PDFs (Khmer & English), chunks them by legal article,
embeds them into a vector database, and serves precise, citation-grounded answers.

## Tech Stack
- **Language**: Python 3.11+
- **PDF Extraction**: PyMuPDF (fitz)
- **Embeddings**: OpenAI `text-embedding-3-large` / `multilingual-e5-large`
- **Vector Store**: PostgreSQL + pgvector
- **Sparse Retrieval**: BM25 (rank-bm25)
- **Reranker**: Cohere Rerank / bge-reranker-large
- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **API**: FastAPI
- **Frontend**: Streamlit (prototype) → React (production)

## Architecture Principles
1. **Clean separation of concerns** — each pipeline stage is an independent module.
2. **Reproducibility** — every step from raw PDF to embedded chunk is versioned and repeatable.
3. **Article-level chunking** — legal hierarchy (Book → Chapter → Article) is preserved as metadata.
4. **Hybrid retrieval** — combine dense (vector) + sparse (BM25) search with reranking.
5. **Strict citation** — every generated answer must cite Law, Chapter, and Article.

## Coding Standards
- Use type hints for all function signatures.
- Write docstrings (Google style) for all public functions and classes.
- Keep functions small and testable (< 50 lines preferred).
- Use `pydantic` models for all data structures passed between modules.
- Environment variables via `.env` files (never commit secrets).

## Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run extraction pipeline
python -m src.pipeline.extract

# Run chunking pipeline
python -m src.pipeline.chunk

# Run embedding pipeline
python -m src.pipeline.embed

# Start API server
uvicorn src.api.main:app --reload

# Start Streamlit UI
streamlit run src.ui/app.py

# Run tests
pytest tests/ -v
```

## Data Flow
```
Raw PDFs → Extract → Clean → Chunk (by Article) → Embed → pgvector
                                                          ↓
User Query → Embed Query → Hybrid Search (Dense + BM25) → Rerank → LLM → Cited Answer
```
