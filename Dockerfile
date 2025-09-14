# ---------- Base image with system libs ----------
FROM python:3.11-slim AS runtime

# Avoid interactive tzdata prompts, speed up apt
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal OS deps needed by onnxruntime (CPU) on Debian/Ubuntu
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ---------- App layer ----------
WORKDIR /app

# Copy only requirement files first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (only what we need)
COPY app ./app
COPY config ./config

# Uvicorn port inside the container
EXPOSE 8080


# Optional: lightweight healthcheck against /healthz
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD \
  wget -qO- http://127.0.0.1:8080/healthz || exit 1

# Start the API. We pass --port 8080 to match EXPOSE; host mapping decides the public port.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
