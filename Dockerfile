# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

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

# ── Copy source (after deps so cache is reused on code-only changes) ──────────
COPY . .

# ── Runtime directories (ensure permissions for non-root user) ──────────────────
RUN mkdir -p uploads outputs && \
    groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]