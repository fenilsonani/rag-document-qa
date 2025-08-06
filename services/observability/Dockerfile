# Observability Service Dockerfile
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
COPY services/observability/requirements.txt ./service_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r service_requirements.txt

# Install observability dependencies
RUN pip install --no-cache-dir \
    prometheus-client>=0.17.0 \
    opentelemetry-api>=1.20.0 \
    opentelemetry-sdk>=1.20.0 \
    opentelemetry-exporter-jaeger-thrift>=1.20.0 \
    opentelemetry-exporter-prometheus>=1.12.0 \
    opentelemetry-instrumentation-fastapi>=0.41b0 \
    opentelemetry-instrumentation-logging>=0.41b0 \
    psutil>=5.9.0

# Copy source code
COPY src/ ./src/
COPY services/ ./services/

# Create necessary directories
RUN mkdir -p /app/logs /app/metrics /app/alerts

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8004 9091

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the service
CMD ["python", "-m", "services.observability.monitoring"]