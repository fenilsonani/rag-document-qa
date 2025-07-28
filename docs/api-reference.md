# API Reference

## EnhancedRAG Class

The main class that orchestrates the RAG pipeline with advanced features.

### Methods

#### `initialize(force_mode=None)`
Initializes the RAG system with online or offline mode.

**Parameters:**
- `force_mode` (str, optional): Force "online" or "offline" mode

**Returns:**
- `dict`: Initialization result with success status and mode

#### `process_documents(uploaded_files, progress_callback=None)`
Processes uploaded documents and creates vector embeddings.

**Parameters:**
- `uploaded_files` (list): List of uploaded file objects
- `progress_callback` (callable, optional): Progress tracking function

**Returns:**
- `dict`: Processing result with success status and metadata

#### `generate_answer(query, conversation_mode=False, context_history=None)`
Generates answers using the RAG pipeline.

**Parameters:**
- `query` (str): User question
- `conversation_mode` (bool): Enable conversation context
- `context_history` (list, optional): Previous conversation history

**Returns:**
- `dict`: Answer with sources and metadata

## DocumentLoader Class

Handles loading and processing of various document formats.

### Supported Formats
- PDF (.pdf)
- Text (.txt)
- Word Documents (.docx)
- Markdown (.md)

### Methods

#### `load_document(file_path)`
Loads a single document from file path.

**Parameters:**
- `file_path` (str): Path to document file

**Returns:**
- `list`: List of Document objects

#### `process_uploaded_files(uploaded_files)`
Processes multiple uploaded files.

**Parameters:**
- `uploaded_files` (list): List of Streamlit uploaded file objects

**Returns:**
- `list`: Combined list of processed documents

## VectorStore Class

Manages ChromaDB vector database operations.

### Methods

#### `add_documents(documents, metadata=None)`
Adds documents to the vector store.

**Parameters:**
- `documents` (list): List of Document objects
- `metadata` (dict, optional): Additional metadata

**Returns:**
- `bool`: Success status

#### `similarity_search(query, k=4, score_threshold=0.7)`
Performs similarity search in the vector store.

**Parameters:**
- `query` (str): Search query
- `k` (int): Number of results to return
- `score_threshold` (float): Minimum similarity score

**Returns:**
- `list`: List of similar documents with scores

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `CHUNK_SIZE` | Document chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap size | 200 |
| `TEMPERATURE` | LLM temperature | 0.7 |
| `MAX_TOKENS` | Maximum tokens per response | 1000 |

### Advanced Configuration

```python
config = Config()
config.chunk_size = 1500
config.chunk_overlap = 300
config.retrieval_k = 6
config.score_threshold = 0.8
```

## Error Handling

### Common Exceptions

- `DocumentProcessingError`: Raised when document processing fails
- `VectorStoreError`: Raised during vector store operations
- `EmbeddingError`: Raised when embedding generation fails
- `APIError`: Raised for API-related issues

### Error Response Format

```python
{
    "success": False,
    "error": "Error message",
    "error_type": "ErrorClassName",
    "details": {...}
}
```