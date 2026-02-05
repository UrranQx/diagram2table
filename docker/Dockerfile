# syntax=docker/dockerfile:1.4
# Diagram2Table Production Dockerfile
# Optimized for VLM inference with GPU support

# Stage 1: Base with Python + CUDA
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12 using ensurepip (distutils-free method)
RUN python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --upgrade pip setuptools wheel

# Create non-root user
RUN useradd -m -u 1000 app
WORKDIR /app

# Stage 2: Dependencies
FROM base AS dependencies

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Stage 3: Production
FROM dependencies AS production

# Build argument for model selection
# TODO: Исправить потом 2B на 3B. И чтобы всегда запускался квантованный вариант модели
ARG MODEL_NAME=qwen2-vl-2b 

# Copy application code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml .
COPY --chown=app:app scripts/ ./scripts/

# Create directories for models, data and HF cache
RUN mkdir -p /app/models /app/data /app/logs /home/app/.cache/huggingface && \
    chown -R app:app /app /home/app/.cache

# Environment variables (set before model download)
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data
ENV LOG_PATH=/app/logs
ENV HF_HOME=/home/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/app/.cache/huggingface/hub
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Pre-download models as root (before switching to app user)
# This caches model weights in the Docker layer for faster production starts
# Downloads only files (no GPU/CUDA required during build)
RUN python scripts/download_models.py --model ${MODEL_NAME} || \
    echo "⚠ Model pre-download failed, will download at runtime"

# Fix cache directory permissions (always runs, even if download failed)
RUN chown -R app:app /home/app/.cache || true

# Switch to non-root user AFTER model download
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Expose ports
EXPOSE 7860 8000

# Default command
CMD ["python", "-m", "src.main", "both"]
