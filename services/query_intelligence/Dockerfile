# Query Intelligence Service Dockerfile
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
COPY services/query_intelligence/requirements.txt ./service_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r service_requirements.txt

# Install NLP dependencies
RUN pip install --no-cache-dir \
    sentence-transformers>=2.6.0 \
    spacy>=3.6.0 \
    transformers>=4.30.0 \
    torch>=2.0.0 \
    accelerate>=0.20.0 \
    nltk>=3.8 \
    scikit-learn>=1.3.0

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY src/ ./src/
COPY services/ ./services/

# Create necessary directories
RUN mkdir -p /app/models /app/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the service
CMD ["python", "-m", "services.query_intelligence.service"]