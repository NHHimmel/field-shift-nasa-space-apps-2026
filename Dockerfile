# Multi-stage Dockerfile for Field Shift (FastAPI + React SPA)
# Compatible with Render, Railway, Hugging Face Spaces, Fly.io, Google Cloud Run

# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/React

COPY React/package*.json ./
RUN npm ci

COPY React/ ./
RUN npm run build

# Stage 2: Python backend with Astral uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runner
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into system Python
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy backend application, database, and scripts
COPY src/ ./src/
COPY FastApi/ ./FastApi/
COPY scripts/ ./scripts/
COPY database/ ./database/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY field_shift_bangladesh.db ./

# Ensure data cache directory exists for NASA weather cache
RUN mkdir -p data/cache

# Copy built frontend from Stage 1 into React/dist
COPY --from=frontend-builder /app/React/dist ./React/dist

# Install the local src package
RUN uv pip install --system --no-cache -e .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Support dynamic port assignment from Render, Railway, Hugging Face ($PORT)
CMD ["sh", "-c", "uvicorn app.main:app --app-dir FastApi --host 0.0.0.0 --port ${PORT:-8000}"]
