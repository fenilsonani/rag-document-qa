# API Gateway Dockerfile
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
COPY services/api_gateway/requirements.txt ./service_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r service_requirements.txt

# Install gateway dependencies
RUN pip install --no-cache-dir \
    httpx>=0.25.0 \
    redis>=5.0.0 \
    pyjwt>=2.8.0 \
    python-multipart>=0.0.6

# Copy source code
COPY src/ ./src/
COPY services/ ./services/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the service
CMD ["python", "-m", "services.api_gateway.gateway"]