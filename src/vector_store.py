"""Vector store implementation using ChromaDB."""

import os
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from .config import Config


class VectorStoreManager:
    """Manages vector store operations using ChromaDB."""
    
    def __init__(self, collection_name: str = "document_embeddings"):
        self.config = Config()
        self.collection_name = collection_name
        self.vector_store_path = Path(self.config.VECTOR_STORE_DIR)
        self.vector_store_path.mkdir(exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.vector_store: Optional[Chroma] = None
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents."""
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")
        
        try:
            # Delete existing collection if it exists
            self.reset_vector_store()
            
            # Create new vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.client,
                collection_name=self.collection_name,
                persist_directory=str(self.vector_store_path)
            )
            
            print(f"Created vector store with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {str(e)}")
    
    def load_vector_store(self) -> Optional[Chroma]:
        """Load existing vector store."""
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                print("No existing vector store found")
                return None
            
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_store_path)
            )
            
            # Check if collection has documents
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            if count == 0:
                print("Vector store exists but is empty")
                return None
            
            print(f"Loaded vector store with {count} documents")
            return self.vector_store
            
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Create or load first.")
        
        if not documents:
            return
        
        try:
            self.vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            raise RuntimeError(f"Failed to add documents: {str(e)}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search on the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata
            )
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Similarity search failed: {str(e)}")
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with relevance scores."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_metadata
            )
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Similarity search with score failed: {str(e)}")
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """Get a retriever interface for the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def reset_vector_store(self) -> None:
        """Reset/delete the vector store."""
        try:
            # Delete collection if it exists
            collections = self.client.list_collections()
            for collection in collections:
                if collection.name == self.collection_name:
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted collection: {self.collection_name}")
                    break
            
            self.vector_store = None
            
        except Exception as e:
            print(f"Error resetting vector store: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        if not self.vector_store:
            return {"exists": False}
        
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "exists": True,
                "name": self.collection_name,
                "document_count": count,
                "embedding_model": self.config.EMBEDDING_MODEL
            }
            
        except Exception as e:
            return {"exists": False, "error": str(e)}
    
    def export_documents(self) -> List[Dict[str, Any]]:
        """Export all documents from the vector store."""
        if not self.vector_store:
            return []
        
        try:
            collection = self.client.get_collection(self.collection_name)
            results = collection.get(include=["documents", "metadatas"])
            
            documents = []
            for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
                documents.append({
                    "id": i,
                    "content": doc,
                    "metadata": metadata
                })
            
            return documents
            
        except Exception as e:
            print(f"Error exporting documents: {str(e)}")
            return []