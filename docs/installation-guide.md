# Installation Guide

## Prerequisites

Before installing the RAG Document Q&A System, ensure you have:

- Python 3.8 or higher
- At least 4GB of RAM (8GB recommended for large documents)
- 2GB of free disk space
- Internet connection for initial setup
- API key from OpenAI or Anthropic

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/fenilsonani/rag-document-qa.git
cd rag-document-qa
```

### 2. Create Virtual Environment

#### Using venv (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using conda
```bash
conda create -n rag-qa python=3.9
conda activate rag-qa
```

### 3. Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Or using pnpm (if you have Node.js):
```bash
pnpm install
```

### 4. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
```env
# Required: At least one API key
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Performance tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TEMPERATURE=0.7
MAX_TOKENS=1000
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Advanced Installation

### Docker Installation

1. Build the Docker image:
```bash
docker build -t rag-document-qa .
```

2. Run the container:
```bash
docker run -p 8501:8501 --env-file .env rag-document-qa
```

### Docker Compose

For a complete setup with services:
```bash
docker-compose up -d
```

## Troubleshooting Installation

### Common Issues

#### 1. Python Version Compatibility
If you encounter Python version issues:
```bash
# Check Python version
python --version

# Install Python 3.9 using pyenv
pyenv install 3.9.16
pyenv local 3.9.16
```

#### 2. Package Installation Errors
For package conflicts:
```bash
# Clear pip cache
pip cache purge

# Upgrade pip
pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

#### 3. ChromaDB Installation Issues
On some systems, ChromaDB might require additional dependencies:
```bash
# On Ubuntu/Debian
sudo apt-get install sqlite3 libsqlite3-dev

# On macOS
brew install sqlite3

# On Windows (use Windows Subsystem for Linux)
```

#### 4. Memory Issues
For systems with limited RAM:
```bash
# Set smaller chunk size
export CHUNK_SIZE=500
export CHUNK_OVERLAP=100

# Or edit .env file
echo "CHUNK_SIZE=500" >> .env
echo "CHUNK_OVERLAP=100" >> .env
```

### Performance Optimization

#### For Better Performance
```env
# Increase chunk size for faster processing
CHUNK_SIZE=1500
CHUNK_OVERLAP=300

# Use fewer retrieval results
RETRIEVAL_K=3
```

#### For Better Accuracy
```env
# Smaller chunks for better precision
CHUNK_SIZE=800
CHUNK_OVERLAP=200

# More retrieval results
RETRIEVAL_K=6
```

## Verification

After installation, verify everything works:

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Upload a test document (try the sample files in `uploads/`)

4. Ask a question about the document

5. Verify you get a response with source citations

## Development Setup

For contributors and developers:

### 1. Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Set up Pre-commit Hooks
```bash
pre-commit install
```

### 3. Run Tests
```bash
python -m pytest tests/
```

### 4. Code Quality Checks
```bash
# Linting
flake8 src/

# Type checking
mypy src/

# Format code
black src/
```

## Next Steps

After successful installation:

1. Read the [User Guide](user-guide.md) for detailed usage instructions
2. Check the [API Reference](api-reference.md) for development
3. Review [Configuration Options](configuration.md) for customization
4. See [Deployment Guide](deployment.md) for production setup