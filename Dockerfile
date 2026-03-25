# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="MediRoute OpenEnv"
LABEL org.opencontainers.image.description="Emergency Department Triage AI Environment"
LABEL org.opencontainers.image.authors="MediRoute Team"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Environment variables (non-secret) ────────────────────────────────────────
ENV MEDIROUTE_SEED=42
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Expose Gradio port (HF Spaces standard) ───────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Launch Gradio app ─────────────────────────────────────────────────────────
CMD ["python", "app.py"]
