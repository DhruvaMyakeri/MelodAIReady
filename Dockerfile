# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Cache-bust label — bump this when requirements or core scripts change ──────
# v5.0 — intelligent arranger rewrite (expert_arranger, to_midi)
LABEL melodai.version="5.0"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ───────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer — rebuild only when requirements.txt changes) ───
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download ResNet18 weights (container has no internet at runtime) ──────
RUN python -c "\
from torchvision.models import resnet18, ResNet18_Weights; \
resnet18(weights=ResNet18_Weights.DEFAULT); \
print('ResNet18 weights cached.')"

# ── Verify critical new dependency ────────────────────────────────────────────
RUN python -c "import ijson; print('ijson OK:', ijson.__version__)"

# ── Copy source (after deps so cache is reused on code-only changes) ──────────
COPY . .

# ── Runtime directories (created fresh on every container start) ──────────────
RUN mkdir -p uploads outputs

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# ── Start ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]