# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set build-time argument for API port
ARG MYAPI_PORT=3014
ARG BAKE_DINOV2_WEIGHTS=true
# Set environment variable for the port
ENV MYAPI_PORT=$MYAPI_PORT
ENV TORCH_HOME=/app/.cache/torch

# Use HTTPS Debian mirrors because some networks block plain HTTP apt traffic.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources

# Install system dependencies including poppler-utils for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-tha \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/share/zoneinfo/Asia/Bangkok /etc/localtime && \
    echo "Asia/Bangkok" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata
    

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-docker.txt .

# Install CPU-only PyTorch first so Docker images do not pull CUDA runtime packages.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    pip install --no-cache-dir --no-deps ultralytics

# Preload DINOv2 only when requested. The project folder is AnomalyDetection;
# this cache is for torch.hub so a future dinov2 active model can run offline.
RUN mkdir -p "$TORCH_HOME" && \
    if [ "$BAKE_DINOV2_WEIGHTS" = "true" ]; then \
        python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True).eval()"; \
    fi

RUN mkdir -p app/uploads app/asset app/detections app/static config logs "$TORCH_HOME"

# Copy application code
COPY app/ ./app
COPY AnomalyDetection/scripts/ ./AnomalyDetection/scripts/
COPY AnomalyDetection/artifacts/models/ ./AnomalyDetection/artifacts/models/
COPY config/.env.example ./config/.env.example
COPY model/ ./model/

RUN groupadd --system appuser && \
    useradd --system --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app


# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp

# Expose port
EXPOSE 3014

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"MYAPI_PORT\", \"3014\")}/version', timeout=5).read()" || exit 1

USER appuser

# Run the application (ไฟล์หลักที่ต้องการรัน)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3014"]

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${MYAPI_PORT}"]
