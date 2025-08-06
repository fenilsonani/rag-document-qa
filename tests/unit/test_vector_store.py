"""
Unit tests for vector_store.py module.
Tests vector store operations, embeddings, similarity search, and document management.
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock ChromaDB and related dependencies before importing
with patch.dict('sys.modules', {
    'chromadb': MagicMock(),
    'chromadb.config': MagicMock(),
    'langchain_community.vectorstores': MagicMock(),
    'langchain_community.embeddings': MagicMock()
}):
    from src.vector_store import VectorStoreManager
    from src.config import Config
    from langchain.schema import Document


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_dir):
    """Create mock Config instance."""
    config = Mock(spec=Config)
    config.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    config.VECTOR_STORE_DIR = temp_dir
    return config


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "doc1.pdf", "page": 1}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "doc2.pdf", "page": 2}
        ),
        Document(
            page_content="Natural language processing enables computers to understand text.",
            metadata={"source": "doc3.pdf", "page": 1}
        )
    ]


@pytest.fixture
def vector_store_manager(mock_config):
    """Create VectorStoreManager instance with mocked dependencies."""
    with patch('src.vector_store.Config', return_value=mock_config), \
         patch('src.vector_store.HuggingFaceEmbeddings') as mock_embeddings, \
         patch('src.vector_store.chromadb.PersistentClient') as mock_client, \
         patch('pathlib.Path.mkdir'):
        
        mock_embeddings.return_value = Mock()
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        manager = VectorStoreManager("test_collection")
        manager.client = mock_client_instance
        
        return manager


class TestVectorStoreManagerInit:
    """Test VectorStoreManager initialization."""
    
    def test_initialization_default_collection(self, mock_config):
        """Test VectorStoreManager initialization with default collection name."""
        with patch('src.vector_store.Config', return_value=mock_config), \
             patch('src.vector_store.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.vector_store.chromadb.PersistentClient') as mock_client, \
             patch('pathlib.Path.mkdir'):
            
            mock_embeddings.return_value = Mock()
            mock_client.return_value = Mock()
            
            manager = VectorStoreManager()
            
            assert manager.collection_name == "document_embeddings"
            assert manager.vector_store is None
            assert hasattr(manager, 'config')
            assert hasattr(manager, 'embeddings')
            assert hasattr(manager, 'client')
    
    def test_initialization_custom_collection(self, mock_config):
        """Test VectorStoreManager initialization with custom collection name."""
        with patch('src.vector_store.Config', return_value=mock_config), \
             patch('src.vector_store.HuggingFaceEmbeddings'), \
             patch('src.vector_store.chromadb.PersistentClient'), \
             patch('pathlib.Path.mkdir'):
            
            manager = VectorStoreManager("custom_collection")
            
            assert manager.collection_name == "custom_collection"
    
    def test_vector_store_path_creation(self, mock_config):
        """Test that vector store directory is created."""
        with patch('src.vector_store.Config', return_value=mock_config), \
             patch('src.vector_store.HuggingFaceEmbeddings'), \
             patch('src.vector_store.chromadb.PersistentClient'), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            VectorStoreManager()
            
            mock_mkdir.assert_called_once_with(exist_ok=True)


class TestCreateVectorStore:
    """Test vector store creation functionality."""
    
    def test_create_vector_store_success(self, vector_store_manager, sample_documents):
        """Test successful vector store creation."""
        with patch('src.vector_store.Chroma.from_documents') as mock_from_docs, \
             patch.object(vector_store_manager, 'reset_vector_store') as mock_reset:
            
            mock_vector_store = Mock()
            mock_from_docs.return_value = mock_vector_store
            
            result = vector_store_manager.create_vector_store(sample_documents)
            
            assert result == mock_vector_store
            assert vector_store_manager.vector_store == mock_vector_store
            mock_reset.assert_called_once()
            mock_from_docs.assert_called_once()
    
    def test_create_vector_store_empty_documents(self, vector_store_manager):
        """Test vector store creation with empty document list."""
        with pytest.raises(ValueError, match="Cannot create vector store from empty document list"):
            vector_store_manager.create_vector_store([])
    
    def test_create_vector_store_exception(self, vector_store_manager, sample_documents):
        """Test vector store creation exception handling."""
        with patch('src.vector_store.Chroma.from_documents', side_effect=Exception("Creation failed")), \
             patch.object(vector_store_manager, 'reset_vector_store'):
            
            with pytest.raises(RuntimeError, match="Failed to create vector store"):
                vector_store_manager.create_vector_store(sample_documents)


class TestLoadVectorStore:
    """Test vector store loading functionality."""
    
    def test_load_vector_store_success(self, vector_store_manager):
        """Test successful vector store loading."""
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 10
        
        vector_store_manager.client.list_collections.return_value = [mock_collection]
        vector_store_manager.client.get_collection.return_value = mock_collection
        
        with patch('src.vector_store.Chroma') as mock_chroma:
            mock_vector_store = Mock()
            mock_chroma.return_value = mock_vector_store
            
            result = vector_store_manager.load_vector_store()
            
            assert result == mock_vector_store
            assert vector_store_manager.vector_store == mock_vector_store
    
    def test_load_vector_store_collection_not_found(self, vector_store_manager):
        """Test loading when collection doesn't exist."""
        mock_collection = Mock()
        mock_collection.name = "different_collection"
        
        vector_store_manager.client.list_collections.return_value = [mock_collection]
        
        result = vector_store_manager.load_vector_store()
        
        assert result is None
        assert vector_store_manager.vector_store is None
    
    def test_load_vector_store_empty_collection(self, vector_store_manager):
        """Test loading empty collection."""
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 0
        
        vector_store_manager.client.list_collections.return_value = [mock_collection]
        vector_store_manager.client.get_collection.return_value = mock_collection
        
        with patch('src.vector_store.Chroma'):
            result = vector_store_manager.load_vector_store()
            
            assert result is None
    
    def test_load_vector_store_exception(self, vector_store_manager):
        """Test vector store loading exception handling."""
        vector_store_manager.client.list_collections.side_effect = Exception("Loading failed")
        
        result = vector_store_manager.load_vector_store()
        
        assert result is None


class TestAddDocuments:
    """Test document addition functionality."""
    
    def test_add_documents_success(self, vector_store_manager, sample_documents):
        """Test successful document addition."""
        mock_vector_store = Mock()
        vector_store_manager.vector_store = mock_vector_store
        
        vector_store_manager.add_documents(sample_documents)
        
        mock_vector_store.add_documents.assert_called_once_with(sample_documents)
    
    def test_add_documents_no_vector_store(self, vector_store_manager, sample_documents):
        """Test adding documents when vector store not initialized."""
        with pytest.raises(ValueError, match="Vector store not initialized"):
            vector_store_manager.add_documents(sample_documents)
    
    def test_add_documents_empty_list(self, vector_store_manager):
        """Test adding empty document list."""
        mock_vector_store = Mock()
        vector_store_manager.vector_store = mock_vector_store
        
        vector_store_manager.add_documents([])
        
        # Should return early without calling add_documents
        mock_vector_store.add_documents.assert_not_called()
    
    def test_add_documents_exception(self, vector_store_manager, sample_documents):
        """Test document addition exception handling."""
        mock_vector_store = Mock()
        mock_vector_store.add_documents.side_effect = Exception("Addition failed")
        vector_store_manager.vector_store = mock_vector_store
        
        with pytest.raises(RuntimeError, match="Failed to add documents"):
            vector_store_manager.add_documents(sample_documents)


class TestSimilaritySearch:
    """Test similarity search functionality."""
    
    def test_similarity_search_success(self, vector_store_manager, sample_documents):
        """Test successful similarity search."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = sample_documents[:2]
        vector_store_manager.vector_store = mock_vector_store
        
        results = vector_store_manager.similarity_search("machine learning", k=2)
        
        assert len(results) == 2
        assert results == sample_documents[:2]
        mock_vector_store.similarity_search.assert_called_once_with(
            query="machine learning",
            k=2,
            filter=None
        )
    
    def test_similarity_search_with_filter(self, vector_store_manager, sample_documents):
        """Test similarity search with metadata filter."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = [sample_documents[0]]
        vector_store_manager.vector_store = mock_vector_store
        
        filter_metadata = {"source": "doc1.pdf"}
        results = vector_store_manager.similarity_search(
            "machine learning", 
            k=4, 
            filter_metadata=filter_metadata
        )
        
        assert len(results) == 1
        mock_vector_store.similarity_search.assert_called_once_with(
            query="machine learning",
            k=4,
            filter=filter_metadata
        )
    
    def test_similarity_search_no_vector_store(self, vector_store_manager):
        """Test similarity search when vector store not initialized."""
        with pytest.raises(ValueError, match="Vector store not initialized"):
            vector_store_manager.similarity_search("test query")
    
    def test_similarity_search_exception(self, vector_store_manager):
        """Test similarity search exception handling."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.side_effect = Exception("Search failed")
        vector_store_manager.vector_store = mock_vector_store
        
        with pytest.raises(RuntimeError, match="Similarity search failed"):
            vector_store_manager.similarity_search("test query")


class TestSimilaritySearchWithScore:
    """Test similarity search with scores functionality."""
    
    def test_similarity_search_with_score_success(self, vector_store_manager, sample_documents):
        """Test successful similarity search with scores."""
        mock_vector_store = Mock()
        mock_results = [(sample_documents[0], 0.9), (sample_documents[1], 0.8)]
        mock_vector_store.similarity_search_with_score.return_value = mock_results
        vector_store_manager.vector_store = mock_vector_store
        
        results = vector_store_manager.similarity_search_with_score("machine learning", k=2)
        
        assert len(results) == 2
        assert results == mock_results
        assert results[0][1] == 0.9  # Check score
        assert results[1][1] == 0.8
    
    def test_similarity_search_with_score_filter(self, vector_store_manager):
        """Test similarity search with score and metadata filter."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.return_value = []
        vector_store_manager.vector_store = mock_vector_store
        
        filter_metadata = {"page": 1}
        vector_store_manager.similarity_search_with_score(
            "test", 
            k=3, 
            filter_metadata=filter_metadata
        )
        
        mock_vector_store.similarity_search_with_score.assert_called_once_with(
            query="test",
            k=3,
            filter=filter_metadata
        )
    
    def test_similarity_search_with_score_no_vector_store(self, vector_store_manager):
        """Test similarity search with score when vector store not initialized."""
        with pytest.raises(ValueError, match="Vector store not initialized"):
            vector_store_manager.similarity_search_with_score("test query")
    
    def test_similarity_search_with_score_exception(self, vector_store_manager):
        """Test similarity search with score exception handling."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.side_effect = Exception("Search failed")
        vector_store_manager.vector_store = mock_vector_store
        
        with pytest.raises(RuntimeError, match="Similarity search with score failed"):
            vector_store_manager.similarity_search_with_score("test query")


class TestRetriever:
    """Test retriever functionality."""
    
    def test_get_retriever_success(self, vector_store_manager):
        """Test successful retriever creation."""
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        vector_store_manager.vector_store = mock_vector_store
        
        retriever = vector_store_manager.get_retriever()
        
        assert retriever == mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
    
    def test_get_retriever_custom_kwargs(self, vector_store_manager):
        """Test retriever creation with custom search kwargs."""
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        vector_store_manager.vector_store = mock_vector_store
        
        search_kwargs = {"k": 10, "filter": {"source": "test.pdf"}}
        retriever = vector_store_manager.get_retriever(search_kwargs)
        
        assert retriever == mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs=search_kwargs)
    
    def test_get_retriever_no_vector_store(self, vector_store_manager):
        """Test retriever creation when vector store not initialized."""
        with pytest.raises(ValueError, match="Vector store not initialized"):
            vector_store_manager.get_retriever()


class TestResetVectorStore:
    """Test vector store reset functionality."""
    
    def test_reset_vector_store_success(self, vector_store_manager):
        """Test successful vector store reset."""
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        
        vector_store_manager.client.list_collections.return_value = [mock_collection]
        vector_store_manager.vector_store = Mock()
        
        vector_store_manager.reset_vector_store()
        
        vector_store_manager.client.delete_collection.assert_called_once_with("test_collection")
        assert vector_store_manager.vector_store is None
    
    def test_reset_vector_store_collection_not_found(self, vector_store_manager):
        """Test reset when collection doesn't exist."""
        mock_collection = Mock()
        mock_collection.name = "different_collection"
        
        vector_store_manager.client.list_collections.return_value = [mock_collection]
        vector_store_manager.vector_store = Mock()
        
        vector_store_manager.reset_vector_store()
        
        vector_store_manager.client.delete_collection.assert_not_called()
        assert vector_store_manager.vector_store is None
    
    def test_reset_vector_store_exception(self, vector_store_manager):
        """Test vector store reset exception handling."""
        vector_store_manager.client.list_collections.side_effect = Exception("Reset failed")
        vector_store_manager.vector_store = Mock()
        
        # Should not raise exception, just log error
        vector_store_manager.reset_vector_store()
        
        # Vector store should still be set to None
        assert vector_store_manager.vector_store is None


class TestCollectionInfo:
    """Test collection information functionality."""
    
    def test_get_collection_info_success(self, vector_store_manager, mock_config):
        """Test successful collection info retrieval."""
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        
        vector_store_manager.client.get_collection.return_value = mock_collection
        vector_store_manager.vector_store = Mock()
        
        info = vector_store_manager.get_collection_info()
        
        expected_info = {
            "exists": True,
            "name": "test_collection",
            "document_count": 42,
            "embedding_model": mock_config.EMBEDDING_MODEL
        }
        
        assert info == expected_info
    
    def test_get_collection_info_no_vector_store(self, vector_store_manager):
        """Test collection info when vector store not initialized."""
        info = vector_store_manager.get_collection_info()
        
        assert info == {"exists": False}
    
    def test_get_collection_info_exception(self, vector_store_manager):
        """Test collection info exception handling."""
        vector_store_manager.client.get_collection.side_effect = Exception("Info failed")
        vector_store_manager.vector_store = Mock()
        
        info = vector_store_manager.get_collection_info()
        
        assert info["exists"] is False
        assert "error" in info
        assert info["error"] == "Info failed"


class TestExportDocuments:
    """Test document export functionality."""
    
    def test_export_documents_success(self, vector_store_manager):
        """Test successful document export."""
        mock_collection = Mock()
        mock_results = {
            "documents": ["Doc content 1", "Doc content 2"],
            "metadatas": [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
        }
        mock_collection.get.return_value = mock_results
        
        vector_store_manager.client.get_collection.return_value = mock_collection
        vector_store_manager.vector_store = Mock()
        
        documents = vector_store_manager.export_documents()
        
        expected_documents = [
            {"id": 0, "content": "Doc content 1", "metadata": {"source": "doc1.pdf"}},
            {"id": 1, "content": "Doc content 2", "metadata": {"source": "doc2.pdf"}}
        ]
        
        assert documents == expected_documents
        mock_collection.get.assert_called_once_with(include=["documents", "metadatas"])
    
    def test_export_documents_no_vector_store(self, vector_store_manager):
        """Test document export when vector store not initialized."""
        documents = vector_store_manager.export_documents()
        
        assert documents == []
    
    def test_export_documents_exception(self, vector_store_manager):
        """Test document export exception handling."""
        vector_store_manager.client.get_collection.side_effect = Exception("Export failed")
        vector_store_manager.vector_store = Mock()
        
        documents = vector_store_manager.export_documents()
        
        assert documents == []


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_initialization_with_missing_config_attributes(self, temp_dir):
        """Test initialization when config is missing attributes."""
        incomplete_config = Mock()
        incomplete_config.VECTOR_STORE_DIR = temp_dir
        # Missing EMBEDDING_MODEL
        
        with patch('src.vector_store.Config', return_value=incomplete_config), \
             patch('src.vector_store.HuggingFaceEmbeddings', side_effect=AttributeError("Missing attribute")), \
             patch('src.vector_store.chromadb.PersistentClient'), \
             patch('pathlib.Path.mkdir'):
            
            with pytest.raises(AttributeError):
                VectorStoreManager()
    
    def test_create_vector_store_with_malformed_documents(self, vector_store_manager):
        """Test vector store creation with malformed documents."""
        malformed_documents = [
            {"not": "a document"},  # Not a Document object
            None,
            "string document"
        ]
        
        with patch('src.vector_store.Chroma.from_documents', side_effect=TypeError("Invalid document type")), \
             patch.object(vector_store_manager, 'reset_vector_store'):
            
            with pytest.raises(RuntimeError):
                vector_store_manager.create_vector_store(malformed_documents)
    
    def test_similarity_search_with_extreme_k_values(self, vector_store_manager):
        """Test similarity search with extreme k values."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = []
        vector_store_manager.vector_store = mock_vector_store
        
        # Very large k value
        results = vector_store_manager.similarity_search("test", k=10000)
        assert isinstance(results, list)
        
        # Zero k value
        results = vector_store_manager.similarity_search("test", k=0)
        assert isinstance(results, list)
        
        # Negative k value
        results = vector_store_manager.similarity_search("test", k=-1)
        assert isinstance(results, list)
    
    def test_operations_with_corrupted_client(self, vector_store_manager):
        """Test operations when ChromaDB client is corrupted."""
        vector_store_manager.client = None
        
        with pytest.raises(AttributeError):
            vector_store_manager.load_vector_store()
    
    def test_concurrent_operations(self, vector_store_manager, sample_documents):
        """Test concurrent vector store operations."""
        mock_vector_store = Mock()
        vector_store_manager.vector_store = mock_vector_store
        
        # Simulate concurrent add and search operations
        vector_store_manager.add_documents(sample_documents[:1])
        mock_vector_store.similarity_search.return_value = sample_documents[:1]
        results = vector_store_manager.similarity_search("test", k=1)
        
        assert len(results) == 1
        mock_vector_store.add_documents.assert_called_once()
        mock_vector_store.similarity_search.assert_called_once()
    
    def test_unicode_and_special_characters(self, vector_store_manager):
        """Test handling of unicode and special characters in queries."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = []
        vector_store_manager.vector_store = mock_vector_store
        
        # Unicode query
        unicode_query = "機械学習について教えて"
        results = vector_store_manager.similarity_search(unicode_query)
        assert isinstance(results, list)
        
        # Special characters
        special_query = "machine learning @#$%^&*()!"
        results = vector_store_manager.similarity_search(special_query)
        assert isinstance(results, list)
        
        # Empty query
        results = vector_store_manager.similarity_search("")
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])