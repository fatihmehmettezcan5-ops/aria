# Aria — single-container Dockerfile for Hugging Face Spaces.
# Runs the FastAPI backend with the built-in HTML chat UI at /.
# Listens on port 7860 (HF Spaces convention) with SQLite by default.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for psycopg, lxml, pdf parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev curl ca-certificates poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (smaller / faster install on Spaces)
COPY requirements.txt /app/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/requirements.txt

# Copy the project
COPY . /app

# HF Spaces sets HOME=/data for persistent storage; we put SQLite there.
ENV HOME=/data \
    DATABASE_URL=sqlite:////data/aria.db \
    ARIA_USE_PGVECTOR=false \
    ARIA_CHECKPOINT=/app/runs/smoke/final.pt \
    ARIA_TOKENIZER=/app/runs/smoke/tokenizer.json \
    ARIA_DEVICE=cpu \
    ALLOWED_ORIGINS=* \
    APP_SECRET=change-me-on-spaces \
    PORT=7860

# Train a tiny smoke model at build time so the Space works immediately.
RUN python -m data.scripts.prepare_smoke \
 && python -m tokenizer.bpe_trainer --config configs/tokenizer_smoke.yaml \
 && python -m training.scripts.train_pretrain --config configs/train_smoke.yaml \
 && python -m training.scripts.train_finetune --config configs/finetune_smoke.yaml \
 && rm -rf data/raw data/processed runs/smoke/pretrain runs/smoke/tb runs/smoke/step_*.pt

# Make /data writable for the SQLite database (HF Spaces grants write access here)
RUN mkdir -p /data && chmod 777 /data

EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers"]
