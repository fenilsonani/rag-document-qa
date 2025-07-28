# Configuration Guide

## Overview

The RAG Document Q&A System can be extensively configured to optimize performance, accuracy, and behavior for your specific use case.

## Environment Variables

### Core Configuration

#### API Keys (Required)
```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4, gpt-4-turbo

# Anthropic Configuration  
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_MODEL=claude-3-sonnet-20240229  # or claude-3-opus-20240229
```

#### Document Processing
```env
# Chunk Configuration
CHUNK_SIZE=1000              # Size of document chunks (tokens)
CHUNK_OVERLAP=200            # Overlap between chunks (tokens)
CHUNK_STRATEGY=recursive     # chunking strategy: recursive, character, token

# Document Limits
MAX_FILE_SIZE=52428800      # Max file size (50MB in bytes)
MAX_FILES_PER_BATCH=10      # Maximum files to process at once
SUPPORTED_EXTENSIONS=.pdf,.txt,.docx,.md  # Comma-separated file extensions
```

#### Vector Store Configuration
```env
# ChromaDB Settings
VECTOR_STORE_PATH=./vector_store     # Path to vector database
COLLECTION_NAME=documents            # ChromaDB collection name
EMBEDDING_MODEL=all-MiniLM-L6-v2    # HuggingFace embedding model
EMBEDDING_DIMENSION=384              # Embedding vector dimension

# Persistence Settings
PERSIST_VECTOR_STORE=true           # Save vectors to disk
AUTO_SAVE_INTERVAL=300              # Auto-save every 5 minutes
```

#### Retrieval Configuration
```env
# Search Parameters
RETRIEVAL_K=4                       # Number of documents to retrieve
SCORE_THRESHOLD=0.7                 # Minimum similarity score
SEARCH_TYPE=similarity              # similarity, mmr, similarity_score_threshold
MMR_DIVERSITY_SCORE=0.5            # Diversity for MMR search

# Reranking
ENABLE_RERANKING=false             # Enable result reranking
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

#### LLM Configuration
```env
# Generation Parameters
TEMPERATURE=0.7                     # Response randomness (0.0-2.0)
MAX_TOKENS=1000                    # Maximum response length
PRESENCE_PENALTY=0.0               # Penalty for repeated topics
FREQUENCY_PENALTY=0.0              # Penalty for repeated words
TOP_P=1.0                          # Nucleus sampling parameter

# Prompt Configuration
SYSTEM_PROMPT_TEMPLATE=default     # Use custom prompt template
INCLUDE_SOURCES=true               # Include source citations
MAX_CONTEXT_LENGTH=4000           # Maximum context window
```

### Performance Tuning

#### Memory Management
```env
# Memory Settings
MAX_MEMORY_USAGE=4096              # Max memory in MB
GARBAGE_COLLECT_FREQUENCY=100      # GC after N operations
CACHE_SIZE=1000                    # LRU cache size for embeddings

# Processing Optimization
BATCH_SIZE=32                      # Batch size for embedding
PARALLEL_PROCESSING=true           # Enable parallel document processing
MAX_WORKERS=4                      # Number of worker threads
```

#### Response Optimization
```env
# Speed vs Quality Tradeoffs
FAST_MODE=false                    # Prioritize speed over accuracy
STREAMING_ENABLED=true             # Enable response streaming
RESPONSE_TIMEOUT=30                # Response timeout in seconds
```

### Advanced Features

#### Conversation Memory
```env
# Memory Configuration
ENABLE_MEMORY=true                 # Enable conversation memory
MEMORY_TYPE=buffer                 # buffer, summary, kg
MAX_MEMORY_TOKENS=2000            # Maximum tokens in memory
MEMORY_DECAY_RATE=0.1             # How quickly to forget old context
```

#### Document Intelligence
```env
# Intelligence Features
ENABLE_DOCUMENT_INSIGHTS=true      # Generate document insights
ENABLE_CROSS_REFERENCE=true       # Cross-reference between documents
ENABLE_SMART_SUGGESTIONS=true     # AI-powered question suggestions
AUTO_CATEGORIZATION=true          # Automatic document categorization
```

#### Query Intelligence
```env
# Query Processing
QUERY_EXPANSION=true              # Expand queries with synonyms
INTENT_DETECTION=true             # Detect user intent
QUERY_CORRECTION=true             # Auto-correct typos
LANGUAGE_DETECTION=auto           # Detect query language
```

## Configuration Files

### Streamlit Configuration

Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 50

[browser]
gatherUsageStats = false
showErrorDetails = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Logging Configuration

Create `logging.yaml`:
```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  detailed:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/rag_system.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: false
    
  langchain:
    level: WARNING
    handlers: [console]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

## Custom Configuration Classes

### Config Class Usage
```python
from src.config import Config

# Load default configuration
config = Config()

# Override specific settings
config.chunk_size = 1500
config.chunk_overlap = 300
config.temperature = 0.5

# Load from custom file
config = Config.from_file('custom_config.json')

# Environment-specific configs
config = Config.for_environment('production')
```

### Custom Prompt Templates

Create `prompts/custom_qa_prompt.txt`:
```text
Context: {context}

Question: {question}

Instructions:
- Provide accurate, detailed answers based solely on the context
- Include specific citations with page numbers when available
- If information is not in the context, clearly state this
- Use a professional, informative tone
- Structure longer answers with bullet points or numbered lists

Answer:
```

Use in configuration:
```env
QA_PROMPT_TEMPLATE=prompts/custom_qa_prompt.txt
```

## Model-Specific Configuration

### OpenAI Models
```env
# GPT-3.5 Turbo (Fast, Cost-effective)
OPENAI_MODEL=gpt-3.5-turbo
MAX_TOKENS=1000
TEMPERATURE=0.7

# GPT-4 (High Quality, Slower)
OPENAI_MODEL=gpt-4
MAX_TOKENS=2000
TEMPERATURE=0.5

# GPT-4 Turbo (Balanced)
OPENAI_MODEL=gpt-4-1106-preview
MAX_TOKENS=1500
TEMPERATURE=0.6
```

### Anthropic Models
```env
# Claude 3 Haiku (Fast)
ANTHROPIC_MODEL=claude-3-haiku-20240307
MAX_TOKENS=1000

# Claude 3 Sonnet (Balanced)
ANTHROPIC_MODEL=claude-3-sonnet-20240229
MAX_TOKENS=1500

# Claude 3 Opus (Highest Quality)
ANTHROPIC_MODEL=claude-3-opus-20240229
MAX_TOKENS=2000
```

### Embedding Models
```env
# Small, Fast Models
EMBEDDING_MODEL=all-MiniLM-L6-v2        # 384 dimensions
EMBEDDING_MODEL=all-MiniLM-L12-v2       # 384 dimensions

# Larger, More Accurate Models
EMBEDDING_MODEL=all-mpnet-base-v2       # 768 dimensions
EMBEDDING_MODEL=all-roberta-large-v1    # 1024 dimensions

# Multilingual Models
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

## Use Case Specific Configurations

### Academic Research
```env
# Optimized for academic papers
CHUNK_SIZE=1200
CHUNK_OVERLAP=300
RETRIEVAL_K=6
SCORE_THRESHOLD=0.75
TEMPERATURE=0.3
ENABLE_CROSS_REFERENCE=true
INCLUDE_SOURCES=true
```

### Business Documents
```env
# Optimized for business reports
CHUNK_SIZE=800
CHUNK_OVERLAP=200
RETRIEVAL_K=4
SCORE_THRESHOLD=0.7
TEMPERATURE=0.5
ENABLE_DOCUMENT_INSIGHTS=true
AUTO_CATEGORIZATION=true
```

### Technical Documentation
```env
# Optimized for technical docs
CHUNK_SIZE=1000
CHUNK_OVERLAP=250
RETRIEVAL_K=5
SCORE_THRESHOLD=0.8
TEMPERATURE=0.2
QUERY_EXPANSION=true
ENABLE_SMART_SUGGESTIONS=true
```

### Legal Documents
```env
# Optimized for legal text
CHUNK_SIZE=1500
CHUNK_OVERLAP=400
RETRIEVAL_K=8
SCORE_THRESHOLD=0.85
TEMPERATURE=0.1
INCLUDE_SOURCES=true
ENABLE_CROSS_REFERENCE=true
```

## Performance Profiles

### Speed-Optimized
```env
CHUNK_SIZE=800
RETRIEVAL_K=3
EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7
FAST_MODE=true
ENABLE_DOCUMENT_INSIGHTS=false
```

### Accuracy-Optimized
```env
CHUNK_SIZE=1200
RETRIEVAL_K=6
EMBEDDING_MODEL=all-mpnet-base-v2
OPENAI_MODEL=gpt-4
TEMPERATURE=0.3
ENABLE_RERANKING=true
ENABLE_CROSS_REFERENCE=true
```

### Balanced
```env
CHUNK_SIZE=1000
RETRIEVAL_K=4
EMBEDDING_MODEL=all-MiniLM-L12-v2
OPENAI_MODEL=gpt-3.5-turbo
TEMPERATURE=0.5
```

## Configuration Validation

### Environment Validation Script
```python
# validate_config.py
import os
import sys
from src.config import Config

def validate_configuration():
    """Validate all configuration settings."""
    config = Config()
    errors = []
    
    # Check required API keys
    if not config.openai_api_key and not config.anthropic_api_key:
        errors.append("At least one API key (OpenAI or Anthropic) is required")
    
    # Validate numeric ranges
    if not (0.0 <= config.temperature <= 2.0):
        errors.append("Temperature must be between 0.0 and 2.0")
    
    if config.chunk_size < 100:
        errors.append("Chunk size should be at least 100")
    
    if config.chunk_overlap >= config.chunk_size:
        errors.append("Chunk overlap must be less than chunk size")
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("Configuration validation passed!")

if __name__ == "__main__":
    validate_configuration()
```

## Dynamic Configuration

### Runtime Configuration Updates
```python
# In your application
def update_config(new_settings):
    """Update configuration at runtime."""
    config = st.session_state.config
    
    for key, value in new_settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Reinitialize components that depend on config
    st.session_state.enhanced_rag.update_config(config)
```

### A/B Testing Configuration
```python
# Feature flags for A/B testing
AB_TEST_CONFIG = {
    'enable_new_chunking': 0.5,      # 50% of users
    'use_advanced_retrieval': 0.3,   # 30% of users
    'enhanced_ui': 0.1               # 10% of users
}
```

## Troubleshooting Configuration

### Common Configuration Issues

1. **API Key Issues:**
```bash
# Test API keys
python -c "import openai; openai.api_key='your-key'; print(openai.Model.list())"
```

2. **Memory Issues:**
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

3. **Model Compatibility:**
```bash
# Test embedding model
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model loaded successfully')
"
```

### Configuration Debugging
```python
# Add to your application for debugging
def print_current_config():
    """Print current configuration for debugging."""
    config = st.session_state.config
    
    st.write("### Current Configuration")
    config_dict = {
        "Chunk Size": config.chunk_size,
        "Chunk Overlap": config.chunk_overlap,
        "Retrieval K": config.retrieval_k,
        "Temperature": config.temperature,
        "Model": config.llm_model,
        "Embedding Model": config.embedding_model
    }
    
    for key, value in config_dict.items():
        st.write(f"**{key}:** {value}")
```