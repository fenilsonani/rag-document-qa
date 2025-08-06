"""
Unit tests for rag_chain.py module.
Tests the main RAG chain functionality, document processing, and question answering.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock langchain before importing
with patch.dict('sys.modules', {
    'langchain.llms': MagicMock(),
    'langchain.embeddings': MagicMock(),
    'langchain.chains': MagicMock(),
    'langchain.prompts': MagicMock(),
    'langchain.schema': MagicMock(),
    'langchain.memory': MagicMock()
}):
    from src.rag_chain import RAGChain
    from src.config import Config
    from langchain.schema import Document


@pytest.fixture
def mock_config():
    """Create mock Config instance."""
    config = Mock(spec=Config)
    config.MODEL_NAME = "test-model"
    config.EMBEDDING_MODEL = "test-embedding"
    config.TEMPERATURE = 0.7
    config.MAX_TOKENS = 1000
    config.CHUNK_SIZE = 1000
    config.CHUNK_OVERLAP = 200
    config.OPENAI_API_KEY = "test-key"
    config.ANTHROPIC_API_KEY = None
    config.VECTOR_STORE_TYPE = "chroma"
    config.get_llm_provider.return_value = "openai"
    return config


@pytest.fixture
def mock_vector_store():
    """Create mock VectorStoreManager."""
    mock_store = Mock()
    mock_store.add_documents.return_value = {"success": True, "count": 5}
    mock_store.search.return_value = [
        ("Document content 1", {"source": "doc1.pdf", "page": 1}, 0.9),
        ("Document content 2", {"source": "doc2.pdf", "page": 2}, 0.8)
    ]
    mock_store.get_stats.return_value = {"total_documents": 10, "total_chunks": 50}
    return mock_store


@pytest.fixture
def mock_document_loader():
    """Create mock DocumentLoader."""
    mock_loader = Mock()
    mock_loader.load_documents.return_value = [
        Document(page_content="Test document content 1", metadata={"source": "test1.pdf"}),
        Document(page_content="Test document content 2", metadata={"source": "test2.pdf"})
    ]
    return mock_loader


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml_guide.pdf", "page": 1}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "dl_intro.pdf", "page": 2}
        )
    ]


class TestRAGChainInitialization:
    """Test RAGChain initialization."""
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')
    @patch('src.rag_chain.DocumentLoader')
    def test_rag_chain_init(self, mock_doc_loader, mock_vector_store, mock_config_class):
        """Test RAGChain initialization."""
        mock_config_class.return_value = mock_config()
        mock_vector_store.return_value = Mock()
        mock_doc_loader.return_value = Mock()
        
        rag_chain = RAGChain()
        
        assert rag_chain.config is not None
        assert rag_chain.vector_store is not None
        assert rag_chain.document_loader is not None
        assert rag_chain.qa_chain is None  # Not initialized yet
        assert rag_chain.conversation_history == []
        assert rag_chain.is_initialized is False
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')  
    @patch('src.rag_chain.DocumentLoader')
    def test_rag_chain_attributes(self, mock_doc_loader, mock_vector_store, mock_config_class):
        """Test RAGChain has required attributes."""
        mock_config_class.return_value = mock_config()
        mock_vector_store.return_value = Mock()
        mock_doc_loader.return_value = Mock()
        
        rag_chain = RAGChain()
        
        # Check all required attributes exist
        required_attrs = [
            'config', 'vector_store', 'document_loader', 'qa_chain',
            'conversation_history', 'is_initialized'
        ]
        
        for attr in required_attrs:
            assert hasattr(rag_chain, attr)


class TestRAGChainInitialize:
    """Test RAG chain initialization method."""
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')
    @patch('src.rag_chain.DocumentLoader')
    @patch('langchain.llms.OpenAI')
    @patch('langchain.embeddings.OpenAIEmbeddings')
    def test_initialize_success_openai(self, mock_embeddings, mock_llm, mock_doc_loader, 
                                     mock_vector_store, mock_config_class):
        """Test successful initialization with OpenAI."""
        # Setup mocks
        config = mock_config()
        config.get_llm_provider.return_value = "openai"
        mock_config_class.return_value = config
        
        mock_vector_store.return_value = Mock()
        mock_doc_loader.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_embeddings.return_value = Mock()
        
        rag_chain = RAGChain()
        
        with patch.object(rag_chain, '_create_qa_chains') as mock_create_chains:
            result = rag_chain.initialize()
            
            assert result["success"] is True
            assert result["provider"] == "openai"
            assert rag_chain.is_initialized is True
            mock_create_chains.assert_called_once()
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')
    @patch('src.rag_chain.DocumentLoader')
    @patch('langchain.llms.Anthropic')
    def test_initialize_success_anthropic(self, mock_llm, mock_doc_loader, 
                                        mock_vector_store, mock_config_class):
        """Test successful initialization with Anthropic."""
        # Setup mocks
        config = mock_config()
        config.get_llm_provider.return_value = "anthropic"
        config.ANTHROPIC_API_KEY = "test-anthropic-key"
        mock_config_class.return_value = config
        
        mock_vector_store.return_value = Mock()
        mock_doc_loader.return_value = Mock()
        mock_llm.return_value = Mock()
        
        rag_chain = RAGChain()
        
        with patch.object(rag_chain, '_create_qa_chains') as mock_create_chains:
            result = rag_chain.initialize()
            
            assert result["success"] is True
            assert result["provider"] == "anthropic"
            assert rag_chain.is_initialized is True
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')
    @patch('src.rag_chain.DocumentLoader')
    def test_initialize_no_api_keys(self, mock_doc_loader, mock_vector_store, mock_config_class):
        """Test initialization failure with no API keys."""
        # Setup mocks with no API keys
        config = mock_config()
        config.OPENAI_API_KEY = None
        config.ANTHROPIC_API_KEY = None
        config.get_llm_provider.return_value = None
        mock_config_class.return_value = config
        
        mock_vector_store.return_value = Mock()
        mock_doc_loader.return_value = Mock()
        
        rag_chain = RAGChain()
        result = rag_chain.initialize()
        
        assert result["success"] is False
        assert "error" in result
        assert rag_chain.is_initialized is False
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')
    @patch('src.rag_chain.DocumentLoader')
    def test_initialize_exception_handling(self, mock_doc_loader, mock_vector_store, mock_config_class):
        """Test initialization handles exceptions."""
        mock_config_class.return_value = mock_config()
        mock_vector_store.side_effect = Exception("Vector store error")
        mock_doc_loader.return_value = Mock()
        
        rag_chain = RAGChain()
        result = rag_chain.initialize()
        
        assert result["success"] is False
        assert "error" in result
        assert rag_chain.is_initialized is False


class TestDocumentProcessing:
    """Test document processing functionality."""
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')
    @patch('src.rag_chain.DocumentLoader')
    def test_process_documents_success(self, mock_doc_loader_class, mock_vector_store_class, mock_config_class):
        """Test successful document processing."""
        # Setup mocks
        mock_config_class.return_value = mock_config()
        
        mock_doc_loader = Mock()
        mock_doc_loader.load_documents.return_value = [
            Document(page_content="Test content", metadata={"source": "test.pdf"})
        ]
        mock_doc_loader_class.return_value = mock_doc_loader
        
        mock_vector_store = Mock()
        mock_vector_store.add_documents.return_value = {"success": True, "count": 1}
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_chain = RAGChain()
        result = rag_chain.process_documents(["test.pdf"])
        
        assert result["success"] is True
        assert result["processed_count"] == 1
        mock_doc_loader.load_documents.assert_called_once_with(["test.pdf"])
        mock_vector_store.add_documents.assert_called_once()
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')
    @patch('src.rag_chain.DocumentLoader')
    def test_process_documents_empty_list(self, mock_doc_loader_class, mock_vector_store_class, mock_config_class):
        """Test processing empty document list."""
        mock_config_class.return_value = mock_config()
        mock_doc_loader_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        rag_chain = RAGChain()
        result = rag_chain.process_documents([])
        
        assert result["success"] is False
        assert "error" in result
    
    @patch('src.rag_chain.Config')
    @patch('src.rag_chain.VectorStoreManager')
    @patch('src.rag_chain.DocumentLoader')
    def test_process_documents_load_failure(self, mock_doc_loader_class, mock_vector_store_class, mock_config_class):
        """Test document processing with load failure."""
        mock_config_class.return_value = mock_config()
        
        mock_doc_loader = Mock()
        mock_doc_loader.load_documents.side_effect = Exception("Load error")
        mock_doc_loader_class.return_value = mock_doc_loader
        
        mock_vector_store_class.return_value = Mock()
        
        rag_chain = RAGChain()
        result = rag_chain.process_documents(["test.pdf"])
        
        assert result["success"] is False
        assert "error" in result


class TestQuestionAnswering:
    """Test question answering functionality."""
    
    def setup_initialized_rag_chain(self):
        """Helper to create an initialized RAG chain."""
        with patch('src.rag_chain.Config') as mock_config_class, \
             patch('src.rag_chain.VectorStoreManager') as mock_vector_store_class, \
             patch('src.rag_chain.DocumentLoader') as mock_doc_loader_class:
            
            mock_config_class.return_value = mock_config()
            mock_vector_store_class.return_value = Mock()
            mock_doc_loader_class.return_value = Mock()
            
            rag_chain = RAGChain()
            rag_chain.is_initialized = True
            rag_chain.qa_chain = Mock()
            
            return rag_chain
    
    def test_ask_question_success(self):
        """Test successful question answering."""
        rag_chain = self.setup_initialized_rag_chain()
        
        # Mock QA chain response
        rag_chain.qa_chain.return_value = {
            "answer": "Machine learning is a subset of AI.",
            "source_documents": [
                Document(page_content="ML content", metadata={"source": "ml.pdf", "page": 1})
            ]
        }
        
        result = rag_chain.ask_question("What is machine learning?")
        
        assert result["success"] is True
        assert "answer" in result
        assert "sources" in result
        assert len(rag_chain.conversation_history) == 1
    
    def test_ask_question_not_initialized(self):
        """Test question asking when not initialized."""
        rag_chain = self.setup_initialized_rag_chain()
        rag_chain.is_initialized = False
        
        result = rag_chain.ask_question("What is ML?")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_ask_question_empty_question(self):
        """Test asking empty question."""
        rag_chain = self.setup_initialized_rag_chain()
        
        result = rag_chain.ask_question("")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_ask_question_with_conversation_memory(self):
        """Test question answering with conversation memory."""
        rag_chain = self.setup_initialized_rag_chain()
        
        # Add existing conversation history
        rag_chain.conversation_history = [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        ]
        
        rag_chain.qa_chain.return_value = {
            "answer": "ML is a subset of AI.",
            "source_documents": []
        }
        
        result = rag_chain.ask_question("What about machine learning?")
        
        assert result["success"] is True
        assert len(rag_chain.conversation_history) == 2
    
    def test_ask_question_qa_chain_exception(self):
        """Test question answering when QA chain raises exception."""
        rag_chain = self.setup_initialized_rag_chain()
        rag_chain.qa_chain.side_effect = Exception("QA error")
        
        result = rag_chain.ask_question("What is ML?")
        
        assert result["success"] is False
        assert "error" in result


class TestSourceFormatting:
    """Test source document formatting."""
    
    def test_format_sources(self):
        """Test formatting of source documents."""
        with patch('src.rag_chain.Config'), \
             patch('src.rag_chain.VectorStoreManager'), \
             patch('src.rag_chain.DocumentLoader'):
            
            rag_chain = RAGChain()
            
            source_docs = [
                Document(
                    page_content="Content 1", 
                    metadata={"source": "doc1.pdf", "page": 1}
                ),
                Document(
                    page_content="Content 2", 
                    metadata={"source": "doc2.pdf", "page": 2}
                )
            ]
            
            formatted = rag_chain._format_sources(source_docs)
            
            assert isinstance(formatted, list)
            assert len(formatted) == 2
            
            for i, source in enumerate(formatted):
                assert "index" in source
                assert "filename" in source
                assert "content" in source
                assert source["index"] == i + 1
    
    def test_format_sources_empty(self):
        """Test formatting empty source list."""
        with patch('src.rag_chain.Config'), \
             patch('src.rag_chain.VectorStoreManager'), \
             patch('src.rag_chain.DocumentLoader'):
            
            rag_chain = RAGChain()
            formatted = rag_chain._format_sources([])
            
            assert isinstance(formatted, list)
            assert len(formatted) == 0


class TestConversationHistory:
    """Test conversation history management."""
    
    def setup_rag_chain(self):
        """Helper to set up RAG chain."""
        with patch('src.rag_chain.Config'), \
             patch('src.rag_chain.VectorStoreManager'), \
             patch('src.rag_chain.DocumentLoader'):
            return RAGChain()
    
    def test_get_conversation_history_empty(self):
        """Test getting empty conversation history."""
        rag_chain = self.setup_rag_chain()
        
        history = rag_chain.get_conversation_history()
        
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_get_conversation_history_with_data(self):
        """Test getting conversation history with data."""
        rag_chain = self.setup_rag_chain()
        
        # Add some conversation history
        rag_chain.conversation_history = [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"}
        ]
        
        history = rag_chain.get_conversation_history()
        
        assert isinstance(history, list)
        assert len(history) == 2
        assert history[0]["question"] == "What is AI?"
    
    def test_clear_conversation_history(self):
        """Test clearing conversation history."""
        rag_chain = self.setup_rag_chain()
        
        # Add some history
        rag_chain.conversation_history = [
            {"question": "What is AI?", "answer": "Artificial Intelligence"}
        ]
        
        result = rag_chain.clear_conversation_history()
        
        assert result["success"] is True
        assert len(rag_chain.conversation_history) == 0


class TestDocumentSearch:
    """Test document search functionality."""
    
    def test_search_documents(self):
        """Test document search."""
        with patch('src.rag_chain.Config') as mock_config_class, \
             patch('src.rag_chain.VectorStoreManager') as mock_vector_store_class, \
             patch('src.rag_chain.DocumentLoader'):
            
            mock_config_class.return_value = mock_config()
            
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = [
                ("Content 1", {"source": "doc1.pdf", "page": 1}, 0.9),
                ("Content 2", {"source": "doc2.pdf", "page": 2}, 0.8)
            ]
            mock_vector_store_class.return_value = mock_vector_store
            
            rag_chain = RAGChain()
            result = rag_chain.search_documents("machine learning", k=2)
            
            assert result["success"] is True
            assert "results" in result
            assert len(result["results"]) == 2
            mock_vector_store.search.assert_called_once_with("machine learning", k=2)
    
    def test_search_documents_exception(self):
        """Test document search with exception."""
        with patch('src.rag_chain.Config') as mock_config_class, \
             patch('src.rag_chain.VectorStoreManager') as mock_vector_store_class, \
             patch('src.rag_chain.DocumentLoader'):
            
            mock_config_class.return_value = mock_config()
            
            mock_vector_store = Mock()
            mock_vector_store.search.side_effect = Exception("Search error")
            mock_vector_store_class.return_value = mock_vector_store
            
            rag_chain = RAGChain()
            result = rag_chain.search_documents("query")
            
            assert result["success"] is False
            assert "error" in result


class TestSystemStatus:
    """Test system status reporting."""
    
    def test_get_system_status(self):
        """Test getting system status."""
        with patch('src.rag_chain.Config') as mock_config_class, \
             patch('src.rag_chain.VectorStoreManager') as mock_vector_store_class, \
             patch('src.rag_chain.DocumentLoader'):
            
            mock_config_class.return_value = mock_config()
            
            mock_vector_store = Mock()
            mock_vector_store.get_stats.return_value = {
                "total_documents": 10,
                "total_chunks": 50
            }
            mock_vector_store_class.return_value = mock_vector_store
            
            rag_chain = RAGChain()
            rag_chain.is_initialized = True
            
            status = rag_chain.get_system_status()
            
            assert isinstance(status, dict)
            assert "is_initialized" in status
            assert "vector_store_stats" in status
            assert "conversation_history_length" in status
            assert status["is_initialized"] is True
    
    def test_get_system_status_not_initialized(self):
        """Test system status when not initialized."""
        with patch('src.rag_chain.Config'), \
             patch('src.rag_chain.VectorStoreManager'), \
             patch('src.rag_chain.DocumentLoader'):
            
            rag_chain = RAGChain()
            status = rag_chain.get_system_status()
            
            assert status["is_initialized"] is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_rag_chain_with_invalid_config(self):
        """Test RAGChain creation with invalid config."""
        with patch('src.rag_chain.Config') as mock_config_class, \
             patch('src.rag_chain.VectorStoreManager'), \
             patch('src.rag_chain.DocumentLoader'):
            
            # Mock config with missing attributes
            mock_config = Mock()
            del mock_config.MODEL_NAME  # Simulate missing attribute
            mock_config_class.return_value = mock_config
            
            # Should not raise exception during initialization
            rag_chain = RAGChain()
            assert rag_chain is not None
    
    def test_ask_question_very_long_question(self):
        """Test asking very long question."""
        rag_chain = self.setup_initialized_rag_chain()
        
        long_question = "What is machine learning? " * 1000  # Very long question
        
        rag_chain.qa_chain.return_value = {
            "answer": "ML is a subset of AI.",
            "source_documents": []
        }
        
        result = rag_chain.ask_question(long_question)
        
        # Should handle long questions gracefully
        assert result["success"] is True
    
    def setup_initialized_rag_chain(self):
        """Helper method for edge case tests."""
        with patch('src.rag_chain.Config') as mock_config_class, \
             patch('src.rag_chain.VectorStoreManager') as mock_vector_store_class, \
             patch('src.rag_chain.DocumentLoader') as mock_doc_loader_class:
            
            mock_config_class.return_value = mock_config()
            mock_vector_store_class.return_value = Mock()
            mock_doc_loader_class.return_value = Mock()
            
            rag_chain = RAGChain()
            rag_chain.is_initialized = True
            rag_chain.qa_chain = Mock()
            
            return rag_chain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])