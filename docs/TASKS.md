# 📋 Implementation Tasks

> Track progress by marking tasks: `[ ]` todo → `[/]` in progress → `[x]` done

---

## Phase 1: Project Setup & Data Collection

### 1.1 Project Scaffolding
- [x] Initialize project directory structure
- [x] Create virtual environment and `requirements.txt`
- [x] Set up `.env.example` with all required config keys
- [x] Set up `.gitignore` for data files, `.env`, and caches
- [ ] Initialize git repository

### 1.2 Data Collection
- [x] Download Civil Code (2007) — English PDF from ODC (300 pages)
- [ ] Download Civil Code (2007) — Khmer PDF from ODC
- [x] Download Law on Commercial Enterprises (2005) — English PDF (scanned)
- [ ] Download Law on Commercial Enterprises (2005) — Khmer PDF
- [ ] Download Amendment to Law on Commercial Enterprises (2022)
- [x] Download Law on Commercial Arbitration (2006) — English PDF (selectable)
- [ ] Download Law on E-Commerce (2019)
- [x] Verify PDF text selectability and scan detection
- [x] Place PDFs in `data/01_raw/en/`

### 1.3 Text Extraction (`src/pipeline/extract.py`)
- [x] Implement `extract_pdf_text()` using PyMuPDF with header/footer clipping
- [x] Implement `detect_scanned_pdf()` to flag image-only PDFs for OCR
- [x] Implement TOC page detection and skipping (Civil Code starts at page 9)
- [x] Strip running headers/footers (e.g., translation disclaimers, page headers)
- [x] Save extracted text to `data/02_extracted/` as `.txt` and `_meta.json` files
- [x] Add logging for extraction progress and errors
- [x] Verify extraction outputs (Civil Code: 774k chars, Commercial Arbitration: 39k chars)

---

## Phase 2: Cleaning & Chunking

### 2.1 Text Cleaning & Formatting
- [x] Remove table of contents / index pages
- [x] Normalize whitespace and line breaks
- [x] Remove decorative characters and formatting artifacts
- [x] Preserve legal numbering (Article, Section, Chapter markers)
- [x] Save cleaned text in `data/02_extracted/`

### 2.2 Legal Chunking (`src/pipeline/chunk.py`)
- [x] Implement Article-level regex chunker for English text
  - Pattern: `r"Article\s+(\d+)[:.]?"` and `r"(\d+)\.\s*(?:\(([^\)]+)\))?"`
- [x] Implement Article-level regex chunker for Khmer text
  - Pattern: `r"មាត្រា\s*([០-៩\d]+)\.?"`
- [x] Extract hierarchical metadata per chunk:
  - `law_name`, `book`, `title`, `chapter`, `section`, `article_number`, `article_title`
- [x] Define `LegalChunk` Pydantic model for validated output
- [x] Implement dominant pattern detection & binary search hierarchy tracking
- [x] Save chunks to `data/04_chunks/` as structured JSON and JSONL (1,347 article chunks)
- [x] Write unit tests with sample articles (8 passed)

---

## Phase 3: Embedding & Storage

### 3.1 Embedding Pipeline (`src/pipeline/embed.py` & `src/infrastructure/ai/openai_embedding.py`)
- [x] Implement `OpenAIEmbedding` adapter using `text-embedding-3-large`
- [x] Add batching logic (batch size 100 chunks per request)
- [x] Implement retry logic with exponential backoff via tenacity
- [x] Add disk caching of embeddings to avoid duplicate API fees
- [x] Implement `python -m src.pipeline.embed` pipeline runner

### 3.2 Database & Storage Setup (`src/infrastructure/storage/`)
- [x] Implement SQLAlchemy ORM `LegalChunkModel` with pgvector column
- [x] Configure HNSW vector index (`vector_cosine_ops`)
- [x] Implement `PgVectorRepository` with PostgreSQL support
- [x] Implement fast local numpy cosine similarity fallback store
- [x] Implement `BM25Retriever` using `rank-bm25` with disk persistence (`bm25_index.pkl`)

---

## Phase 4: Retrieval & Generation

### 4.1 Hybrid Retrieval (`src/application/use_cases/hybrid_retrieve.py`)
- [x] Implement Reciprocal Rank Fusion (RRF) to merge dense + sparse results
- [x] Implement graceful fallback to BM25 when dense vectors are offline
- [x] Add metadata filtering (by law name, chapter)
- [x] Write unit tests for RRF scoring logic (passed)

### 4.2 Reranking (`src/infrastructure/retrieval/cross_encoder_reranker.py`)
- [x] Implement Cross-Encoder reranker adapter (`BAAI/bge-reranker-large` / sentence-transformers)
- [x] Accept top-K from hybrid search, return reranked top-N
- [x] Add score-based fallback if model is not loaded locally

### 4.3 LLM Generation (`src/application/use_cases/answer_legal_qa.py` & `src/infrastructure/ai/openai_llm.py`)
- [x] Design legal system prompt enforcing mandatory article citations
- [x] Implement `OpenAILLM` adapter with GPT-4o / GPT-4o-mini
- [x] Implement citation parsing and verification against retrieved context articles
- [x] Write unit tests with verified & unverified citation detection (passed)

---

## Phase 5: API & User Interface

### 5.1 FastAPI Backend (`src/interfaces/api/`)
- [x] Implement `/api/v1/retrieve` endpoint: hybrid search + reranking
- [x] Implement `/api/v1/qa` endpoint: grounded question answering with verified citations
- [x] Implement `/health` liveness endpoint
- [x] Implement dependency injection wiring (`dependencies.py`)
- [x] Add CORS middleware and Pydantic v2 schemas
- [x] Write API integration tests (`tests/integration/test_api_routes.py` - passed)

### 5.2 Streamlit UI (`src/interfaces/ui/app.py`)
- [x] Build interactive Legal Q&A Assistant tab with example questions
- [x] Build Statutory Article Explorer tab with live keyword & article search
- [x] Display answers with verified/unverified citation badges
- [x] Display expandable source article cards with statutory hierarchy and relevance scores
- [x] Add sidebar statute filters (Civil Code, Commercial Arbitration, All Laws)

---

## Phase 6: Evaluation & Optimization

### 6.1 Evaluation Pipeline (`src/evaluation/`)
- [ ] Create golden Q&A dataset (50+ question-answer-article triples)
- [ ] Implement evaluation using RAGAS metrics:
  - Context Precision, Context Recall, Faithfulness, Answer Relevancy
- [ ] Generate evaluation report with scores per metric
- [ ] Identify and log failure cases (missed articles, hallucinations)

### 6.2 Optimization
- [ ] Tune chunk overlap and size for edge-case articles
- [ ] Experiment with different embedding models
- [ ] Tune hybrid search weights (dense vs. sparse ratio)
- [ ] Optimize reranker top-K and threshold
- [ ] Profile and optimize query latency

---

## Phase 7: Khmer Language Support (Stretch)

### 7.1 Khmer Text Processing
- [ ] Integrate Khmer word segmenter (`khmercut` or `khmersegment`)
- [ ] Test multilingual embeddings (`multilingual-e5-large`) on Khmer text
- [ ] Implement OCR pipeline for scanned Khmer PDFs (Tesseract `khm`)
- [ ] Validate retrieval quality on Khmer-language queries

### 7.2 Bilingual Support
- [ ] Cross-language retrieval: query in English, retrieve Khmer articles (and vice versa)
- [ ] Language detection for incoming queries
- [ ] Parallel article display (English + Khmer side by side)
