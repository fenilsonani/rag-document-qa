# Deployment Guide

## Overview

This guide covers deploying the RAG Document Q&A System in various environments, from local development to production-scale deployments.

## Deployment Options

### 1. Streamlit Cloud (Recommended for Simple Deployments)

#### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account
- API keys for OpenAI/Anthropic

#### Steps
1. **Push to GitHub:**
```bash
git add .
git commit -m "feat: prepare for streamlit cloud deployment"
git push origin main
```

2. **Connect to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Choose `app.py` as the main file

3. **Configure Secrets:**
   In Streamlit Cloud dashboard, add these secrets:
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
CHUNK_SIZE = "1000"
CHUNK_OVERLAP = "200"
```

4. **Deploy:**
   - Click "Deploy"
   - Wait for build to complete
   - Your app will be available at `https://[app-name].streamlit.app`

### 2. Docker Deployment

#### Build Docker Image
```bash
# Build the image
docker build -t rag-document-qa:latest .

# Run locally for testing
docker run -p 8501:8501 --env-file .env rag-document-qa:latest
```

#### Docker Compose (Production Setup)
```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - CHUNK_SIZE=1000
      - CHUNK_OVERLAP=200
    volumes:
      - ./uploads:/app/uploads
      - ./vector_store:/app/vector_store
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-app
    restart: unless-stopped
```

#### Start with Docker Compose
```bash
docker-compose up -d
```

### 3. AWS Deployment

#### Using AWS EC2

1. **Launch EC2 Instance:**
```bash
# Connect to instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Update system
sudo yum update -y
sudo yum install -y docker git

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user
```

2. **Deploy Application:**
```bash
# Clone repository
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa

# Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Build and run
docker build -t rag-app .
docker run -d -p 8501:8501 --env-file .env rag-app
```

3. **Configure Security Group:**
   - Allow inbound traffic on port 8501
   - Configure HTTPS if needed

#### Using AWS ECS (Fargate)

1. **Create Task Definition:**
```json
{
  "family": "rag-document-qa",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "rag-app",
      "image": "your-ecr-repo/rag-document-qa:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-api-key"
        }
      ]
    }
  ]
}
```

2. **Create Service:**
```bash
aws ecs create-service \
  --cluster your-cluster \
  --service-name rag-document-qa \
  --task-definition rag-document-qa:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

### 4. Google Cloud Platform

#### Using Cloud Run

1. **Build and Push to Container Registry:**
```bash
# Configure gcloud
gcloud auth configure-docker

# Build and tag
docker build -t gcr.io/your-project-id/rag-document-qa .
docker push gcr.io/your-project-id/rag-document-qa
```

2. **Deploy to Cloud Run:**
```bash
gcloud run deploy rag-document-qa \
  --image gcr.io/your-project-id/rag-document-qa \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your-key,ANTHROPIC_API_KEY=your-key
```

### 5. Azure Deployment

#### Using Azure Container Instances

```bash
# Create resource group
az group create --name rag-rg --location eastus

# Deploy container
az container create \
  --resource-group rag-rg \
  --name rag-document-qa \
  --image your-registry/rag-document-qa:latest \
  --dns-name-label rag-qa-unique \
  --ports 8501 \
  --environment-variables OPENAI_API_KEY=your-key ANTHROPIC_API_KEY=your-key
```

## Production Configuration

### Environment Variables
```bash
# Production settings
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Performance settings
export CHUNK_SIZE=1500
export CHUNK_OVERLAP=300
export MAX_TOKENS=2000
export TEMPERATURE=0.5
```

### SSL/HTTPS Configuration

#### Nginx Configuration
```nginx
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://rag-app:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Database Persistence

#### Using External Vector Store
```python
# For production, consider using managed vector databases
# PostgreSQL with pgvector
DATABASE_URL = "postgresql://user:pass@host:5432/db"

# Pinecone (managed)
PINECONE_API_KEY = "your-pinecone-key"
PINECONE_ENVIRONMENT = "your-environment"

# Weaviate (managed)
WEAVIATE_URL = "https://your-cluster.weaviate.network"
WEAVIATE_API_KEY = "your-api-key"
```

### Monitoring and Logging

#### Application Monitoring
```python
# Add to app.py
import logging
import structlog
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration')

# Structured logging
logger = structlog.get_logger()
```

#### Health Checks
```python
# health_check.py
import requests
import sys

def health_check():
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    sys.exit(0 if health_check() else 1)
```

### Scaling Considerations

#### Horizontal Scaling
- Use load balancers (Nginx, AWS ALB, GCP Load Balancer)
- Stateless application design
- Shared vector store across instances
- Session affinity for conversation mode

#### Resource Requirements

**Minimum Production Setup:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB SSD
- Network: 1Gbps

**Recommended Production Setup:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 10Gbps

**High-Load Setup:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- Network: 10Gbps+

### Security Best Practices

#### API Key Management
```bash
# Use secrets management services
# AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id openai-api-key

# Azure Key Vault
az keyvault secret show --name openai-api-key --vault-name your-vault

# Google Secret Manager
gcloud secrets versions access latest --secret="openai-api-key"
```

#### Network Security
- Use VPC/Private networks
- Implement WAF (Web Application Firewall)
- Enable HTTPS only
- Configure CORS properly
- Use authentication middleware

### Backup Strategy

#### Vector Store Backup
```bash
# Backup ChromaDB
tar -czf vector_store_backup_$(date +%Y%m%d).tar.gz vector_store/

# Upload to cloud storage
aws s3 cp vector_store_backup_*.tar.gz s3://your-backup-bucket/
```

#### Automated Backups
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="rag_backup_$DATE"

# Create backup
docker exec rag-app tar -czf /tmp/$BACKUP_NAME.tar.gz vector_store/ uploads/

# Copy from container
docker cp rag-app:/tmp/$BACKUP_NAME.tar.gz ./backups/

# Upload to cloud
aws s3 cp ./backups/$BACKUP_NAME.tar.gz s3://your-backup-bucket/

# Cleanup old backups (keep last 30 days)
find ./backups/ -name "rag_backup_*.tar.gz" -mtime +30 -delete
```

### CI/CD Pipeline

#### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t rag-document-qa:${{ github.sha }} .
      
      - name: Deploy to production
        run: |
          # Your deployment commands here
          echo "Deploying to production..."
```

## Troubleshooting Production Issues

### Common Issues

1. **Memory Issues:**
```bash
# Monitor memory usage
docker stats rag-app

# Increase memory limits
docker run -m 4g rag-document-qa
```

2. **Performance Issues:**
```bash
# Check CPU usage
htop

# Optimize chunk size
export CHUNK_SIZE=800
```

3. **Connection Issues:**
```bash
# Check network connectivity
curl -f http://localhost:8501/_stcore/health

# Check logs
docker logs rag-app
```

### Maintenance

#### Regular Updates
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Rebuild Docker image
docker build -t rag-document-qa:latest .

# Rolling update
docker-compose up -d --no-deps --build rag-app
```

#### Performance Monitoring
```bash
# Monitor application metrics
curl http://localhost:8501/metrics

# Database performance
docker exec rag-app sqlite3 vector_store/chroma.sqlite3 ".schema"
```