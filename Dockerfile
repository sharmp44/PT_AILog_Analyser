# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.14-slim

# ── System dependencies ───────────────────────────────────────────────────────
# Required by weasyprint / xhtml2pdf / lxml / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker caches this layer
# (re-runs only if requirements.txt changes, not on every code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# ── Streamlit config ──────────────────────────────────────────────────────────
# Tells Streamlit not to open a browser and to listen on all interfaces
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# ── DuckDB temp path ──────────────────────────────────────────────────────────
# Pipeline writes DuckDB to /tmp — always available in containers
ENV DUCKDB_PATH=/tmp/pt_pipeline.duckdb

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Run the app ───────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
