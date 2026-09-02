# ⚖️ RAG Cambodia Law

> A production-grade **Retrieval-Augmented Generation (RAG)** system for Cambodian Civil & Commercial Law, delivering precise, citation-grounded answers backed by official statutory text.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Project Overview

This system ingests official Cambodian legal statutes, chunks them at the **article level**, and serves legally grounded Q&A through a hybrid retrieval pipeline. Every answer is backed by verified citations to specific law articles.

**Ingested Corpus (1,347 Articles)**
| Statute | Articles |
|---------|----------|
| Civil Code of Cambodia (2007) | 1,297 |
| Law on Commercial Arbitration (2006) | 50 |

---

## 🏗️ Architecture

```
Raw PDFs → Extract → Clean → Chunk (Article-level) → Embed → pgvector + BM25
                                                                      ↓
User Query → Embed Query → Hybrid Search (Dense + BM25) → Rerank → LLM → Cited Answer
```

**Clean Architecture layers:**
- **Domain** — Entities (`LegalChunk`, `Citation`, `QAResponse`) and port interfaces
- **Application** — Use cases (`HybridRetrieveUseCase`, `AnswerLegalQAUseCase`)
- **Infrastructure** — Adapters (PyMuPDF, BM25, OpenAI, pgvector, CrossEncoder)
- **Interfaces** — FastAPI REST API + Streamlit UI

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/rag_cambodia_law.git
cd rag_cambodia_law
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the Pipeline

```bash
# Download official PDFs
python -m src.pipeline.download

# Extract text
python -m src.pipeline.extract

# Chunk into articles
python -m src.pipeline.chunk

# Build BM25 index (+ dense embeddings if API key is set)
python -m src.pipeline.embed
```

### 4. Launch the App

```bash
# Web UI (recommended)
streamlit run src/interfaces/ui/app.py
# → http://localhost:8501

# REST API
uvicorn src.interfaces.api.main:app --reload
# → http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

```bash
docker compose up --build
```

Services:
- **UI**: http://localhost:8501
- **API**: http://localhost:8000/docs
- **PostgreSQL + pgvector**: localhost:5432

---

## 📊 Evaluation

```bash
# BM25-only metrics (no API key needed)
python -m src.evaluation.evaluate_rag

# Full RAGAS metrics (requires OPENAI_API_KEY)
python -m src.evaluation.evaluate_rag --use-llm
```

Target metrics:
| Metric | Target |
|--------|--------|
| Hit Rate @5 | ≥ 80% |
| Context Recall | ≥ 0.80 |
| Faithfulness | ≥ 0.90 |
| Answer Relevancy | ≥ 0.85 |

---

## 🧪 Tests

```bash
pytest tests/ -v
# 15 tests across unit + integration
```

---

## 📁 Project Structure

```
rag_cambodia_law/
├── data/
│   ├── 01_raw/en/          # Raw legal PDFs
│   ├── 02_extracted/       # Extracted text
│   ├── 04_chunks/          # Article-level JSON chunks
│   └── indices/            # BM25 + dense vector indices
├── src/
│   ├── domain/             # Entities, ports, exceptions
│   ├── application/        # Use cases and DTOs
│   ├── infrastructure/     # Adapters (PyMuPDF, BM25, OpenAI, pgvector)
│   ├── interfaces/         # FastAPI + Streamlit
│   ├── pipeline/           # CLI pipeline runners
│   └── evaluation/         # RAGAS evaluation scripts
└── tests/
    ├── unit/               # Unit tests
    ├── integration/        # API integration tests
    └── evaluation/         # Golden Q&A dataset
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Extraction | PyMuPDF |
| Sparse Retrieval | BM25 (rank-bm25) |
| Dense Embeddings | OpenAI `text-embedding-3-large` |
| Vector Database | PostgreSQL + pgvector |
| Reranking | Cross-Encoder (`BAAI/bge-reranker-large`) |
| LLM | OpenAI GPT-4o |
| API | FastAPI |
| UI | Streamlit |
| Evaluation | RAGAS |

---

## ⚠️ Legal Disclaimer

This system is a **research and educational tool**. It does not constitute legal advice. Always consult a qualified legal professional for advice on specific legal matters.

---

## 📄 Data Sources

Legal texts sourced from [Open Development Cambodia (ODC)](https://data.opendevelopmentcambodia.net/laws_record).
