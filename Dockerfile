# Multi-stage production build for Autonomous Tech Radar & Just-In-Time Learning Agent
FROM python:3.12-slim AS base

# System dependencies for WeasyPrint, Pango, HarfBuzz, CFFI, and networking
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    fonts-dejavu-core \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create artifact directories
RUN mkdir -p /app/artifacts/pdfs && chmod -R 777 /app/artifacts

# Copy application source code
COPY . /app

# Non-root user for security best practices
RUN useradd -m -u 1000 techradar && chown -R techradar:techradar /app
USER techradar

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
