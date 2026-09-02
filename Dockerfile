# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install OS-level dependencies for PyMuPDF and pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    tesseract-ocr \
    tesseract-ocr-khm \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only runtime OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    tesseract-ocr \
    tesseract-ocr-khm \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . .

# Create data directories
RUN mkdir -p data/01_raw/en data/02_extracted data/03_cleaned data/04_chunks data/indices

EXPOSE 8000 8501

# Default: start API server
CMD ["uvicorn", "src.interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
