"""Embedding and document processing service."""

from typing import List, Dict, Any, Optional
import time

from langchain.schema import Document
from .document_loader import DocumentProcessor
from .vector_store import VectorStoreManager
from .config import Config


class EmbeddingService:
    """Service for processing documents and creating embeddings."""
    
    def __init__(self):
        self.config = Config()
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize the service by loading existing vector store or preparing for new one."""
        try:
            # Try to load existing vector store
            existing_store = self.vector_store_manager.load_vector_store()
            
            if existing_store:
                self.is_initialized = True
                print("Initialized with existing vector store")
                return True
            else:
                print("No existing vector store found - ready to process new documents")
                return True
                
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            return False
    
    def process_and_embed_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process documents and create embeddings."""
        start_time = time.time()
        
        try:
            # Validate files
            valid_files = []
            invalid_files = []
            
            for file_path in file_paths:
                if self.document_processor.validate_file(file_path):
                    valid_files.append(file_path)
                else:
                    invalid_files.append(file_path)
            
            if not valid_files:
                return {
                    "success": False,
                    "error": "No valid files found",
                    "invalid_files": invalid_files
                }
            
            # Process documents
            print(f"Processing {len(valid_files)} valid files...")
            documents = self.document_processor.process_documents(valid_files)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents could be processed",
                    "invalid_files": invalid_files
                }
            
            # Create embeddings and store in vector database
            print("Creating embeddings and storing in vector database...")
            self.vector_store_manager.create_vector_store(documents)
            self.is_initialized = True
            
            processing_time = time.time() - start_time
            
            # Get collection info
            collection_info = self.vector_store_manager.get_collection_info()
            
            return {
                "success": True,
                "processed_files": len(valid_files),
                "invalid_files": invalid_files,
                "total_chunks": len(documents),
                "processing_time": round(processing_time, 2),
                "collection_info": collection_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def add_documents_to_existing_store(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add new documents to existing vector store."""
        if not self.is_initialized:
            return {
                "success": False,
                "error": "Vector store not initialized. Process documents first."
            }
        
        start_time = time.time()
        
        try:
            # Process new documents
            documents = self.document_processor.process_documents(file_paths)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents could be processed"
                }
            
            # Add to existing vector store
            self.vector_store_manager.add_documents(documents)
            
            processing_time = time.time() - start_time
            collection_info = self.vector_store_manager.get_collection_info()
            
            return {
                "success": True,
                "added_chunks": len(documents),
                "processing_time": round(processing_time, 2),
                "collection_info": collection_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to add documents: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def search_documents(
        self, 
        query: str, 
        k: int = 4, 
        include_scores: bool = False
    ) -> Dict[str, Any]:
        """Search for relevant documents."""
        if not self.is_initialized:
            return {
                "success": False,
                "error": "Vector store not initialized"
            }
        
        try:
            if include_scores:
                results = self.vector_store_manager.similarity_search_with_score(query, k)
                formatted_results = []
                
                for doc, score in results:
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(score)
                    })
            else:
                results = self.vector_store_manager.similarity_search(query, k)
                formatted_results = []
                
                for doc in results:
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "result_count": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Search failed: {str(e)}"
            }
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """Get retriever for RAG chain."""
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store_manager.get_retriever(search_kwargs)
    
    def reset_store(self) -> Dict[str, Any]:
        """Reset the vector store."""
        try:
            self.vector_store_manager.reset_vector_store()
            self.is_initialized = False
            
            return {
                "success": True,
                "message": "Vector store reset successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Reset failed: {str(e)}"
            }
    
    def get_store_status(self) -> Dict[str, Any]:
        """Get current status of the vector store."""
        collection_info = self.vector_store_manager.get_collection_info()
        
        return {
            "initialized": self.is_initialized,
            "collection_info": collection_info,
            "embedding_model": self.config.EMBEDDING_MODEL,
            "chunk_size": self.config.CHUNK_SIZE,
            "chunk_overlap": self.config.CHUNK_OVERLAP
        }
    
    def export_all_documents(self) -> List[Dict[str, Any]]:
        """Export all documents from the vector store."""
        if not self.is_initialized:
            return []
        
        return self.vector_store_manager.export_documents()