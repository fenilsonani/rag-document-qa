"""Document loading and preprocessing functionality."""

import os
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import Config


class DocumentProcessor:
    """Handles document loading, processing, and chunking."""
    
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on its file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")
        
        try:
            if extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif extension == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(f"No loader available for {extension}")
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "filename": path.name,
                    "file_type": extension
                })
            
            return documents
            
        except Exception as e:
            raise RuntimeError(f"Error loading document {file_path}: {str(e)}")
    
    def load_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """Load multiple documents."""
        all_documents: List[Document] = []
        
        for file_path in file_paths:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {str(e)}")
                continue
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval."""
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Complete document processing pipeline."""
        # Load documents
        documents = self.load_multiple_documents(file_paths)
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        print(f"Processed {len(documents)} documents into {len(chunks)} chunks")
        
        return chunks
    
    def get_uploaded_files(self, upload_dir: Optional[str] = None) -> List[str]:
        """Get list of uploaded files."""
        upload_path = Path(upload_dir or self.config.UPLOAD_DIR)
        
        if not upload_path.exists():
            return []
        
        files = []
        for file_path in upload_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.config.SUPPORTED_EXTENSIONS:
                files.append(str(file_path))
        
        return files
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if a file can be processed."""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False
        
        # Check file extension
        if path.suffix.lower() not in self.config.SUPPORTED_EXTENSIONS:
            return False
        
        # Check file size (limit to 50MB)
        if path.stat().st_size > 50 * 1024 * 1024:
            return False
        
        return True