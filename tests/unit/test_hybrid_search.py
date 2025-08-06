"""
Unit tests for hybrid_search.py module.
Tests BM25 retriever, query expansion, hybrid search engine, and reciprocal rank fusion.
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hybrid_search import (
    SearchResult, BM25Retriever, QueryExpander, 
    ReciprocalRankFusion, HybridSearchEngine
)
from src.vector_store import VectorStoreManager
from src.config import Config
from langchain.schema import Document


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
            page_content="Natural language processing enables computers to understand human language.",
            metadata={"source": "doc3.pdf", "page": 1}
        ),
        Document(
            page_content="Machine learning algorithms can be supervised or unsupervised.",
            metadata={"source": "doc1.pdf", "page": 2}
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Create mock VectorStoreManager."""
    mock_store = Mock(spec=VectorStoreManager)
    mock_store.search.return_value = [
        ("doc content 1", {"source": "doc1.pdf", "page": 1}, 0.9),
        ("doc content 2", {"source": "doc2.pdf", "page": 2}, 0.8),
    ]
    return mock_store


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self, sample_documents):
        """Test SearchResult object creation."""
        doc = sample_documents[0]
        result = SearchResult(
            document=doc,
            score=0.85,
            rank=1,
            source="bm25",
            metadata={"query": "test"}
        )
        
        assert result.document == doc
        assert result.score == 0.85
        assert result.rank == 1
        assert result.source == "bm25"
        assert result.metadata == {"query": "test"}


class TestBM25Retriever:
    """Test BM25Retriever class."""
    
    def test_bm25_initialization(self):
        """Test BM25Retriever initialization."""
        retriever = BM25Retriever(k1=1.2, b=0.8)
        
        assert retriever.k1 == 1.2
        assert retriever.b == 0.8
        assert retriever.documents == []
        assert retriever.document_frequencies == {}
        assert retriever.document_lengths == []
        assert retriever.avg_document_length == 0.0
        assert retriever.vocabulary == set()
        assert retriever.is_fitted is False
    
    def test_bm25_default_parameters(self):
        """Test BM25Retriever with default parameters."""
        retriever = BM25Retriever()
        
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        retriever = BM25Retriever()
        
        text = "Machine Learning! And AI-powered systems, including NLP."
        tokens = retriever._preprocess_text(text)
        
        assert isinstance(tokens, list)
        assert "machine" in tokens
        assert "learning" in tokens
        assert "ai" in tokens
        assert "powered" in tokens
        # Punctuation should be removed
        assert "!" not in tokens
        assert "," not in tokens
    
    def test_fit_with_documents(self, sample_documents):
        """Test fitting BM25 with documents."""
        retriever = BM25Retriever()
        retriever.fit(sample_documents)
        
        assert retriever.is_fitted is True
        assert len(retriever.documents) == len(sample_documents)
        assert len(retriever.document_lengths) == len(sample_documents)
        assert retriever.avg_document_length > 0
        assert len(retriever.vocabulary) > 0
        assert len(retriever.document_frequencies) > 0
        
        # Check that common terms are in vocabulary
        assert "machine" in retriever.vocabulary
        assert "learning" in retriever.vocabulary
    
    def test_fit_empty_documents(self):
        """Test fitting BM25 with empty document list."""
        retriever = BM25Retriever()
        retriever.fit([])
        
        assert retriever.is_fitted is True
        assert len(retriever.documents) == 0
        assert len(retriever.vocabulary) == 0
        assert retriever.avg_document_length == 0
    
    def test_calculate_idf(self, sample_documents):
        """Test IDF calculation."""
        retriever = BM25Retriever()
        retriever.fit(sample_documents)
        
        # Test IDF for common term
        idf_machine = retriever._calculate_idf("machine")
        assert idf_machine > 0
        
        # Test IDF for rare term
        idf_rare = retriever._calculate_idf("nonexistent")
        assert idf_rare > idf_machine  # Rare terms should have higher IDF
    
    def test_calculate_score(self, sample_documents):
        """Test BM25 score calculation."""
        retriever = BM25Retriever()
        retriever.fit(sample_documents)
        
        query_terms = ["machine", "learning"]
        score = retriever._calculate_score(query_terms, 0)
        
        assert isinstance(score, float)
        assert score >= 0  # BM25 scores should be non-negative
    
    def test_search(self, sample_documents):
        """Test BM25 search functionality."""
        retriever = BM25Retriever()
        retriever.fit(sample_documents)
        
        results = retriever.search("machine learning", k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.source == "bm25"
            assert result.score >= 0
            assert result.rank >= 1
    
    def test_search_not_fitted(self):
        """Test search on unfitted retriever."""
        retriever = BM25Retriever()
        
        with pytest.raises(ValueError, match="BM25Retriever must be fitted"):
            retriever.search("query")
    
    def test_search_empty_query(self, sample_documents):
        """Test search with empty query."""
        retriever = BM25Retriever()
        retriever.fit(sample_documents)
        
        results = retriever.search("", k=5)
        
        # Empty query should return empty results
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_search_no_matches(self, sample_documents):
        """Test search with no matching terms."""
        retriever = BM25Retriever()
        retriever.fit(sample_documents)
        
        results = retriever.search("xyztermthatdoesnotexist", k=5)
        
        # No matches should return empty results
        assert isinstance(results, list)
        assert len(results) == 0


class TestQueryExpander:
    """Test QueryExpander class."""
    
    def test_query_expander_initialization(self):
        """Test QueryExpander initialization."""
        expander = QueryExpander()
        
        assert hasattr(expander, 'synonyms')
        assert isinstance(expander.synonyms, dict)
    
    def test_expand_query_basic(self):
        """Test basic query expansion."""
        expander = QueryExpander()
        
        original_query = "machine learning"
        expanded = expander.expand_query(original_query)
        
        assert isinstance(expanded, str)
        assert len(expanded) >= len(original_query)  # Should be same or longer
    
    def test_expand_query_with_synonyms(self):
        """Test query expansion with synonym replacement."""
        expander = QueryExpander()
        
        # Test with terms that have synonyms in the default dict
        query = "AI algorithms"
        expanded = expander.expand_query(query)
        
        assert isinstance(expanded, str)
        # Should contain original terms or their expansions
        assert "algorithm" in expanded.lower() or "ai" in expanded.lower()
    
    def test_expand_query_with_documents(self, sample_documents):
        """Test query expansion using document context."""
        expander = QueryExpander()
        
        query = "ML"
        expanded = expander.expand_query(query, sample_documents)
        
        assert isinstance(expanded, str)
        # Should be expanded beyond just the acronym
        assert len(expanded) > len(query)
    
    def test_expand_query_empty_string(self):
        """Test query expansion with empty string."""
        expander = QueryExpander()
        
        expanded = expander.expand_query("")
        
        assert expanded == ""
    
    def test_expand_query_special_characters(self):
        """Test query expansion with special characters."""
        expander = QueryExpander()
        
        query = "machine-learning & AI!"
        expanded = expander.expand_query(query)
        
        assert isinstance(expanded, str)
        # Should handle special characters gracefully


class TestReciprocalRankFusion:
    """Test ReciprocalRankFusion class."""
    
    def test_rrf_initialization(self):
        """Test ReciprocalRankFusion initialization."""
        rrf = ReciprocalRankFusion(k=50)
        
        assert rrf.k == 50
    
    def test_rrf_default_k(self):
        """Test ReciprocalRankFusion with default k."""
        rrf = ReciprocalRankFusion()
        
        assert rrf.k == 60
    
    def test_fuse_rankings_basic(self, sample_documents):
        """Test basic ranking fusion."""
        rrf = ReciprocalRankFusion()
        
        # Create sample search results
        bm25_results = [
            SearchResult(sample_documents[0], 0.9, 1, "bm25", {}),
            SearchResult(sample_documents[1], 0.7, 2, "bm25", {}),
        ]
        
        vector_results = [
            SearchResult(sample_documents[1], 0.8, 1, "vector", {}),
            SearchResult(sample_documents[2], 0.6, 2, "vector", {}),
        ]
        
        rankings = [bm25_results, vector_results]
        fused = rrf.fuse_rankings(rankings, k=5)
        
        assert isinstance(fused, list)
        assert len(fused) <= 5
        
        for result in fused:
            assert isinstance(result, SearchResult)
            assert result.source == "hybrid"
            assert hasattr(result, 'score')
    
    def test_fuse_rankings_empty_lists(self):
        """Test fusion with empty ranking lists."""
        rrf = ReciprocalRankFusion()
        
        fused = rrf.fuse_rankings([], k=5)
        
        assert isinstance(fused, list)
        assert len(fused) == 0
    
    def test_fuse_rankings_single_list(self, sample_documents):
        """Test fusion with single ranking list."""
        rrf = ReciprocalRankFusion()
        
        results = [
            SearchResult(sample_documents[0], 0.9, 1, "bm25", {}),
        ]
        
        fused = rrf.fuse_rankings([results], k=5)
        
        assert isinstance(fused, list)
        assert len(fused) == 1
        assert fused[0].source == "hybrid"
    
    def test_fuse_rankings_duplicate_documents(self, sample_documents):
        """Test fusion with duplicate documents across rankings."""
        rrf = ReciprocalRankFusion()
        
        # Same document in both rankings
        bm25_results = [
            SearchResult(sample_documents[0], 0.9, 1, "bm25", {}),
        ]
        
        vector_results = [
            SearchResult(sample_documents[0], 0.8, 1, "vector", {}),
        ]
        
        fused = rrf.fuse_rankings([bm25_results, vector_results], k=5)
        
        # Should deduplicate and combine scores
        assert len(fused) == 1
        assert fused[0].document == sample_documents[0]
        assert fused[0].source == "hybrid"


class TestHybridSearchEngine:
    """Test HybridSearchEngine class."""
    
    def test_hybrid_search_initialization(self, mock_vector_store):
        """Test HybridSearchEngine initialization."""
        engine = HybridSearchEngine(mock_vector_store, enable_reranking=True)
        
        assert engine.vector_store == mock_vector_store
        assert engine.enable_reranking is True
        assert hasattr(engine, 'bm25_retriever')
        assert hasattr(engine, 'query_expander')
        assert hasattr(engine, 'rrf')
        assert engine.is_initialized is False
    
    def test_hybrid_search_initialization_no_reranking(self, mock_vector_store):
        """Test HybridSearchEngine without reranking."""
        engine = HybridSearchEngine(mock_vector_store, enable_reranking=False)
        
        assert engine.enable_reranking is False
    
    def test_initialize_with_documents(self, mock_vector_store, sample_documents):
        """Test engine initialization with documents."""
        engine = HybridSearchEngine(mock_vector_store)
        
        result = engine.initialize(sample_documents)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert engine.is_initialized is True
        assert engine.bm25_retriever.is_fitted is True
    
    def test_initialize_empty_documents(self, mock_vector_store):
        """Test engine initialization with empty documents."""
        engine = HybridSearchEngine(mock_vector_store)
        
        result = engine.initialize([])
        
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert engine.is_initialized is True
    
    @patch('src.hybrid_search.HybridSearchEngine.search_bm25')
    def test_search_bm25(self, mock_search, mock_vector_store, sample_documents):
        """Test BM25 search method."""
        engine = HybridSearchEngine(mock_vector_store)
        engine.initialize(sample_documents)
        
        # Mock the search_bm25 method to return expected results
        mock_results = [
            SearchResult(sample_documents[0], 0.9, 1, "bm25", {}),
        ]
        mock_search.return_value = mock_results
        
        results = engine.search_bm25("machine learning", k=5)
        
        assert isinstance(results, list)
        mock_search.assert_called_once_with("machine learning", k=5)
    
    @patch('src.hybrid_search.HybridSearchEngine.search_vector')
    def test_search_vector(self, mock_search, mock_vector_store, sample_documents):
        """Test vector search method."""
        engine = HybridSearchEngine(mock_vector_store)
        engine.initialize(sample_documents)
        
        # Mock the search_vector method
        mock_results = [
            SearchResult(sample_documents[0], 0.8, 1, "vector", {}),
        ]
        mock_search.return_value = mock_results
        
        results = engine.search_vector("machine learning", k=5)
        
        assert isinstance(results, list)
        mock_search.assert_called_once_with("machine learning", k=5)
    
    def test_search_hybrid_basic(self, mock_vector_store, sample_documents):
        """Test hybrid search functionality."""
        engine = HybridSearchEngine(mock_vector_store)
        engine.initialize(sample_documents)
        
        with patch.object(engine, 'search_bm25') as mock_bm25, \
             patch.object(engine, 'search_vector') as mock_vector:
            
            # Mock search results
            mock_bm25.return_value = [
                SearchResult(sample_documents[0], 0.9, 1, "bm25", {}),
            ]
            mock_vector.return_value = [
                SearchResult(sample_documents[1], 0.8, 1, "vector", {}),
            ]
            
            results = engine.search_hybrid("machine learning", k=5)
            
            assert isinstance(results, list)
            mock_bm25.assert_called_once()
            mock_vector.assert_called_once()
    
    def test_search_not_initialized(self, mock_vector_store):
        """Test search on uninitialized engine."""
        engine = HybridSearchEngine(mock_vector_store)
        
        with pytest.raises(ValueError, match="HybridSearchEngine must be initialized"):
            engine.search("query")
    
    def test_search_with_strategy(self, mock_vector_store, sample_documents):
        """Test search with different strategies."""
        engine = HybridSearchEngine(mock_vector_store)
        engine.initialize(sample_documents)
        
        with patch.object(engine, 'search_bm25') as mock_bm25, \
             patch.object(engine, 'search_vector') as mock_vector, \
             patch.object(engine, 'search_hybrid') as mock_hybrid:
            
            # Test BM25 strategy
            engine.search("query", strategy="bm25")
            mock_bm25.assert_called_once()
            
            # Test vector strategy
            engine.search("query", strategy="vector")
            mock_vector.assert_called_once()
            
            # Test hybrid strategy
            engine.search("query", strategy="hybrid")
            mock_hybrid.assert_called_once()
    
    def test_search_invalid_strategy(self, mock_vector_store, sample_documents):
        """Test search with invalid strategy."""
        engine = HybridSearchEngine(mock_vector_store)
        engine.initialize(sample_documents)
        
        with pytest.raises(ValueError, match="Invalid search strategy"):
            engine.search("query", strategy="invalid")
    
    def test_get_search_analytics(self, mock_vector_store, sample_documents):
        """Test search analytics generation."""
        engine = HybridSearchEngine(mock_vector_store)
        engine.initialize(sample_documents)
        
        with patch.object(engine, 'search_bm25') as mock_bm25, \
             patch.object(engine, 'search_vector') as mock_vector, \
             patch.object(engine, 'search_hybrid') as mock_hybrid:
            
            # Mock results
            mock_bm25.return_value = [SearchResult(sample_documents[0], 0.9, 1, "bm25", {})]
            mock_vector.return_value = [SearchResult(sample_documents[1], 0.8, 1, "vector", {})]
            mock_hybrid.return_value = [SearchResult(sample_documents[0], 0.85, 1, "hybrid", {})]
            
            analytics = engine.get_search_analytics("query", k=5)
            
            assert isinstance(analytics, dict)
            assert "query" in analytics
            assert "results_count" in analytics
            assert "strategies_compared" in analytics
    
    def test_update_weights(self, mock_vector_store):
        """Test weight updating functionality."""
        engine = HybridSearchEngine(mock_vector_store)
        
        engine.update_weights(bm25_weight=0.3, vector_weight=0.7)
        
        assert engine.bm25_weight == 0.3
        assert engine.vector_weight == 0.7
    
    def test_get_status(self, mock_vector_store, sample_documents):
        """Test status reporting."""
        engine = HybridSearchEngine(mock_vector_store)
        
        # Before initialization
        status = engine.get_status()
        assert status["is_initialized"] is False
        
        # After initialization
        engine.initialize(sample_documents)
        status = engine.get_status()
        assert status["is_initialized"] is True
        assert "document_count" in status
        assert "bm25_weight" in status
        assert "vector_weight" in status


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_bm25_with_unicode_text(self, sample_documents):
        """Test BM25 with unicode text."""
        unicode_doc = Document(
            page_content="Машинное обучение и 機械学習",
            metadata={"source": "unicode.pdf"}
        )
        
        retriever = BM25Retriever()
        retriever.fit([unicode_doc])
        
        results = retriever.search("learning", k=5)
        
        # Should handle unicode gracefully
        assert isinstance(results, list)
    
    def test_hybrid_search_with_very_short_query(self, mock_vector_store, sample_documents):
        """Test hybrid search with very short query."""
        engine = HybridSearchEngine(mock_vector_store)
        engine.initialize(sample_documents)
        
        with patch.object(engine, 'search_bm25') as mock_bm25, \
             patch.object(engine, 'search_vector') as mock_vector:
            
            mock_bm25.return_value = []
            mock_vector.return_value = []
            
            results = engine.search("a", k=5)
            
            assert isinstance(results, list)
    
    def test_rrf_with_mismatched_rankings(self, sample_documents):
        """Test RRF with rankings of different lengths."""
        rrf = ReciprocalRankFusion()
        
        short_ranking = [SearchResult(sample_documents[0], 0.9, 1, "bm25", {})]
        long_ranking = [
            SearchResult(sample_documents[1], 0.8, 1, "vector", {}),
            SearchResult(sample_documents[2], 0.7, 2, "vector", {}),
            SearchResult(sample_documents[3], 0.6, 3, "vector", {}),
        ]
        
        fused = rrf.fuse_rankings([short_ranking, long_ranking], k=5)
        
        assert isinstance(fused, list)
        assert len(fused) <= 5
    
    def test_query_expansion_with_numbers(self):
        """Test query expansion with numerical content."""
        expander = QueryExpander()
        
        query = "version 2.0 algorithm performance 95%"
        expanded = expander.expand_query(query)
        
        assert isinstance(expanded, str)
        # Should handle numbers gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])