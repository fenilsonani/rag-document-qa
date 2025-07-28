"""Configuration settings for the RAG application."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration class."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    # Document processing settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # LLM settings
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    
    # Paths
    UPLOAD_DIR: str = "uploads"
    VECTOR_STORE_DIR: str = "vector_store"
    
    # Supported file types
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx", ".md"]
    
    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM Models (Latest versions as of 2025)
    OPENAI_MODEL: str = "gpt-4o"  # GPT-4o (flagship multimodal model)
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"  # Claude 4.0 Sonnet
    
    @classmethod
    def validate_api_keys(cls) -> bool:
        """Check if at least one API key is configured."""
        return bool(cls.OPENAI_API_KEY or cls.ANTHROPIC_API_KEY)
    
    @classmethod
    def get_available_llm_provider(cls) -> Optional[str]:
        """Return the first available LLM provider."""
        if cls.OPENAI_API_KEY:
            return "openai"
        elif cls.ANTHROPIC_API_KEY:
            return "anthropic"
        return None