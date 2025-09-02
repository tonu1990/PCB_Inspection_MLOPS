# ---- Base: Python slim on Debian Bookworm (good on Pi OS 64-bit)
FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (minimal). libgomp for some ORT kernels, libjpeg/z for Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libjpeg62-turbo libpng16-16 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser
WORKDIR /app

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ app/

# Switch to non-root
USER appuser

EXPOSE 8080
ENV HOST=0.0.0.0 PORT=8080

# MODEL_PATH comes from host env-file; default is the standard late-binding path
ENV MODEL_PATH=/opt/edge/models/current.onnx

# Accept an app image tag at build time; fallback to "dev"
ARG APP_IMAGE_TAG=dev
ENV APP_IMAGE_TAG=${APP_IMAGE_TAG}

# Start the API
CMD ["bash", "-lc", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]
