# 🎯 Project Goal

## Vision
Build a **production-quality RAG system** for Cambodian Civil & Commercial Law that enables
users (lawyers, businesses, students, researchers) to ask natural-language questions and
receive **precise, article-cited answers** grounded in official legal documents.

---

## Target Legal Documents

| Document | Year | Coverage | Priority |
|----------|------|----------|----------|
| Civil Code of Cambodia (ក្រមរដ្ឋប្បវេណី) | 2007 | Obligations, Contracts, Property, Torts | 🔴 P0 |
| Law on Commercial Enterprises (ច្បាប់ស្តីពីសហគ្រាសពាណិជ្ជកម្ម) | 2005 | Partnerships, LLCs, PLCs, Shareholders | 🔴 P0 |
| Amendment to Law on Commercial Enterprises | 2022 | Digital registration, single-member LLC | 🟡 P1 |
| Law on Commercial Arbitration | 2006 | Business dispute resolution | 🟡 P1 |
| Law on E-Commerce | 2019 | Digital contracts, e-signatures | 🟢 P2 |

---

## Success Criteria

### Functional Requirements
- [ ] Ingest PDF documents (both Khmer and English) with OCR fallback
- [ ] Chunk by legal hierarchy: Book → Chapter → Section → Article
- [ ] Hybrid retrieval: dense (vector) + sparse (BM25) search
- [ ] Rerank retrieved results using a cross-encoder
- [ ] Generate answers with **mandatory article citations**
- [ ] Web interface for querying

### Quality Requirements
- [ ] **Context Precision** ≥ 0.85 — retrieved articles are relevant to the query
- [ ] **Context Recall** ≥ 0.80 — all relevant articles are retrieved
- [ ] **Faithfulness** ≥ 0.90 — generated answers don't hallucinate beyond context
- [ ] **Answer Relevancy** ≥ 0.85 — answers directly address the query

### Non-Functional Requirements
- [ ] Query response time < 5 seconds (end-to-end)
- [ ] Modular architecture — each pipeline stage independently testable
- [ ] Reproducible pipeline — re-run from raw PDFs to embedded chunks deterministically
- [ ] Environment-based configuration (no hardcoded secrets)

---

## Key Technical Decisions

### 1. Article-Level Chunking (not fixed-length)
Legal texts have a strict hierarchy. Cutting at arbitrary character limits would split
conditions, exceptions, and cross-references. We chunk at the **Article** level with
parent metadata (Book, Chapter, Section) attached.

### 2. Hybrid Search (Dense + Sparse)
- **Dense** (vector similarity): Handles semantic queries like *"What happens if a partner leaves?"*
- **Sparse** (BM25 keyword): Handles exact lookups like *"Article 144"* or *"statute of limitations"*
- **Fusion**: Reciprocal Rank Fusion (RRF) to merge results before reranking.

### 3. PostgreSQL + pgvector (not FAISS/Chroma)
- Single database for vectors, metadata, and full-text search (BM25 via `tsvector`)
- ACID-compliant, production-ready, no extra infrastructure
- Native SQL for complex filtered queries (e.g., "only Civil Code, Book 4")

### 4. Strict Citation Enforcement
The LLM system prompt mandates citing specific Law → Chapter → Article for every claim.
Answers without citations are considered failures. This prevents legal hallucinations.

---

## Milestones

| Phase | Milestone | Target |
|-------|-----------|--------|
| **Phase 1** | Data collection & extraction pipeline | Week 1 |
| **Phase 2** | Chunking, embedding, vector storage | Week 1–2 |
| **Phase 3** | Retrieval pipeline (hybrid + rerank) | Week 2 |
| **Phase 4** | LLM generation with citations | Week 2–3 |
| **Phase 5** | Streamlit UI & evaluation (RAGAS) | Week 3 |
| **Phase 6** | Khmer language support & OCR | Week 4+ |
