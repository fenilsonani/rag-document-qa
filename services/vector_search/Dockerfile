# Vector Search Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY services/vector_search/requirements.txt ./service_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r service_requirements.txt

# Install vector search dependencies
RUN pip install --no-cache-dir \
    chromadb>=0.4.0 \
    faiss-cpu>=1.7.0 \
    sentence-transformers>=2.6.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    transformers>=4.30.0 \
    torch>=2.0.0

# Copy source code
COPY src/ ./src/
COPY services/ ./services/

# Create necessary directories
RUN mkdir -p /app/vector_store /app/models /app/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the service
CMD ["python", "-m", "services.vector_search.service"]