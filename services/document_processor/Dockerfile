# Document Processing Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    tesseract-ocr \
    poppler-utils \
    libopencv-dev \
    python3-opencv \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY services/document_processor/requirements.txt ./service_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r service_requirements.txt

# Install additional PDF processing dependencies
RUN pip install --no-cache-dir \
    pdfplumber>=0.9.0 \
    camelot-py[cv]>=0.10.1 \
    PyMuPDF>=1.23.0 \
    tabula-py>=2.8.0 \
    pillow>=10.0.0 \
    opencv-python>=4.8.0 \
    pytesseract>=0.3.10 \
    transformers>=4.30.0 \
    torch>=2.0.0 \
    accelerate>=0.20.0

# Copy source code
COPY src/ ./src/
COPY services/ ./services/

# Create necessary directories
RUN mkdir -p /app/uploads /app/models /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the service
CMD ["python", "-m", "services.document_processor.service"]