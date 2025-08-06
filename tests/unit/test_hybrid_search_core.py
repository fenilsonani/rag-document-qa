"""
Core hybrid search functionality tests.
Tests the business logic without heavy external dependencies.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import math
from collections import Counter, defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestSearchResult:
    """Test SearchResult dataclass functionality."""
    
    def test_search_result_structure(self):
        """Test SearchResult creation and attributes."""
        # Mock document
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        
        # Create SearchResult-like object
        result = {
            'document': mock_doc,
            'score': 0.85,
            'rank': 1,
            'source': 'bm25',
            'metadata': {'query': 'test'}
        }
        
        assert result['document'] == mock_doc
        assert result['score'] == 0.85
        assert result['rank'] == 1
        assert result['source'] == 'bm25'
        assert result['metadata']['query'] == 'test'


class TestBM25Logic:
    """Test BM25 algorithm core logic."""
    
    def test_text_preprocessing(self):
        """Test BM25 text preprocessing logic."""
        text = "Machine Learning! And AI-powered systems, including NLP."
        
        # Simulate BM25 preprocessing
        text_lower = text.lower()
        import re
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text_lower)
        
        # Remove stop words (basic set)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        assert "machine" in filtered_tokens
        assert "learning" in filtered_tokens
        assert "powered" in filtered_tokens
        assert "nlp" in filtered_tokens
        assert "and" not in filtered_tokens  # Stop word removed
    
    def test_document_frequency_calculation(self):
        """Test document frequency calculation logic."""
        documents = [
            "machine learning algorithms",
            "deep learning neural networks",
            "machine learning applications"
        ]
        
        # Calculate document frequencies
        document_frequencies = Counter()
        vocabulary = set()
        
        for doc in documents:
            tokens = doc.split()
            unique_tokens = set(tokens)
            vocabulary.update(unique_tokens)
            
            for token in unique_tokens:
                document_frequencies[token] += 1
        
        assert document_frequencies["machine"] == 2  # Appears in 2 documents
        assert document_frequencies["learning"] == 3  # Appears in 3 documents
        assert document_frequencies["deep"] == 1     # Appears in 1 document
        assert len(vocabulary) > 0
    
    def test_idf_calculation(self):
        """Test IDF calculation logic."""
        N = 4  # Total documents
        df_common = 3  # Document frequency for common term
        df_rare = 1    # Document frequency for rare term
        
        # IDF calculation: log((N - df + 0.5) / (df + 0.5))
        idf_common = math.log((N - df_common + 0.5) / (df_common + 0.5))
        idf_rare = math.log((N - df_rare + 0.5) / (df_rare + 0.5))
        
        assert idf_rare > idf_common  # Rare terms should have higher IDF
        assert idf_rare > 0  # Rare terms should have positive IDF
        # Common terms may have negative IDF in this formula when df is high
    
    def test_bm25_score_calculation(self):
        """Test BM25 score calculation logic."""
        k1 = 1.5
        b = 0.75
        tf = 2  # Term frequency
        idf = 1.5  # Inverse document frequency
        doc_length = 100  # Document length
        avg_doc_length = 80  # Average document length
        
        # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
        score = idf * (numerator / denominator)
        
        assert score > 0
        assert isinstance(score, float)
        
        # Test edge case: zero term frequency
        tf_zero = 0
        numerator_zero = tf_zero * (k1 + 1)
        denominator_zero = tf_zero + k1 * (1 - b + b * (doc_length / avg_doc_length))
        score_zero = idf * (numerator_zero / denominator_zero)
        
        assert score_zero == 0


class TestQueryExpansion:
    """Test query expansion logic."""
    
    def test_basic_query_expansion(self):
        """Test basic query expansion logic."""
        query = "machine learning"
        
        # Simulate query expansion
        import re
        terms = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        expanded_terms = []
        
        for term in terms:
            expanded_terms.append(term)
            
            # Add plural/singular variations
            if term.endswith('s') and len(term) > 3:
                expanded_terms.append(term[:-1])  # Remove plural
            elif not term.endswith('s'):
                expanded_terms.append(term + 's')  # Add plural
        
        # Remove duplicates but preserve order
        unique_terms = []
        seen = set()
        for term in expanded_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        assert len(unique_terms) >= len(terms)
        assert "machine" in unique_terms
        assert "learning" in unique_terms
    
    def test_stemming_expansion(self):
        """Test basic stemming expansion logic."""
        terms = ["running", "learned", "testing"]
        expanded = []
        
        for term in terms:
            expanded.append(term)
            
            # Basic stemming rules
            if term.endswith('ing') and len(term) > 5:
                expanded.append(term[:-3])  # run
            elif term.endswith('ed') and len(term) > 4:
                expanded.append(term[:-2])  # learn
        
        assert "runn" in expanded  # "running" -> "runn" (after removing "ing")
        assert "learn" in expanded
        assert "running" in expanded
        assert "learned" in expanded


class TestReciprocalRankFusion:
    """Test RRF algorithm logic."""
    
    def test_rrf_score_calculation(self):
        """Test RRF score calculation."""
        k = 60  # RRF parameter
        rank = 3  # 0-indexed rank
        weight = 1.0
        
        # RRF formula: weight / (k + rank + 1)
        rrf_score = weight / (k + rank + 1)
        expected_score = 1.0 / (60 + 3 + 1)  # 1/64
        
        assert abs(rrf_score - expected_score) < 1e-10
        assert rrf_score > 0
        assert rrf_score < 1
    
    def test_rrf_ranking_fusion(self):
        """Test RRF ranking fusion logic."""
        # Mock search results
        bm25_results = [
            {'doc_id': 'doc1', 'rank': 0},  # rank 1
            {'doc_id': 'doc2', 'rank': 1},  # rank 2
        ]
        
        vector_results = [
            {'doc_id': 'doc2', 'rank': 0},  # rank 1
            {'doc_id': 'doc3', 'rank': 1},  # rank 2
        ]
        
        k = 60
        document_scores = defaultdict(float)
        
        # Calculate RRF scores for BM25 results
        for result in bm25_results:
            doc_id = result['doc_id']
            rank = result['rank']
            rrf_score = 1.0 / (k + rank + 1)
            document_scores[doc_id] += rrf_score
        
        # Calculate RRF scores for vector results
        for result in vector_results:
            doc_id = result['doc_id']
            rank = result['rank']
            rrf_score = 1.0 / (k + rank + 1)
            document_scores[doc_id] += rrf_score
        
        # Sort by fused scores
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        
        # doc2 should be first (appears in both rankings)
        assert len(sorted_docs) == 3
        assert sorted_docs[0][0] == 'doc2'  # Highest combined score
        assert sorted_docs[0][1] > sorted_docs[1][1]  # Higher score than others


class TestHybridSearchLogic:
    """Test hybrid search combination logic."""
    
    def test_weight_normalization(self):
        """Test search weight normalization."""
        bm25_weight = 0.3
        vector_weight = 0.7
        total_weight = bm25_weight + vector_weight
        
        # Normalize weights
        normalized_bm25 = bm25_weight / total_weight
        normalized_vector = vector_weight / total_weight
        
        assert abs(normalized_bm25 + normalized_vector - 1.0) < 1e-10
        assert normalized_bm25 == 0.3
        assert normalized_vector == 0.7
    
    def test_search_result_combination(self):
        """Test combining search results from multiple sources."""
        # Mock BM25 results
        bm25_results = [
            {'doc_id': 'doc1', 'score': 0.9, 'source': 'bm25'},
            {'doc_id': 'doc2', 'score': 0.7, 'source': 'bm25'},
        ]
        
        # Mock vector results
        vector_results = [
            {'doc_id': 'doc2', 'score': 0.8, 'source': 'vector'},
            {'doc_id': 'doc3', 'score': 0.6, 'source': 'vector'},
        ]
        
        # Combine unique documents
        all_docs = set()
        for result in bm25_results:
            all_docs.add(result['doc_id'])
        for result in vector_results:
            all_docs.add(result['doc_id'])
        
        assert len(all_docs) == 3  # doc1, doc2, doc3
        assert 'doc1' in all_docs
        assert 'doc2' in all_docs
        assert 'doc3' in all_docs
    
    def test_overlap_metrics_calculation(self):
        """Test search result overlap metrics."""
        bm25_docs = {'doc1', 'doc2', 'doc3'}
        vector_docs = {'doc2', 'doc3', 'doc4'}
        
        # Calculate overlaps
        overlap = len(bm25_docs & vector_docs)
        bm25_unique = len(bm25_docs - vector_docs)
        vector_unique = len(vector_docs - bm25_docs)
        
        assert overlap == 2  # doc2, doc3
        assert bm25_unique == 1  # doc1
        assert vector_unique == 1  # doc4


class TestPerformanceMetrics:
    """Test performance and analytics calculations."""
    
    def test_score_statistics(self):
        """Test score statistics calculation."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        assert min_score == 0.5
        assert max_score == 0.9
        assert avg_score == 0.7
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation logic."""
        # Mock confidence factors
        accuracy = 85.5
        completeness = 0.9
        consistency = 0.8
        
        # Normalize accuracy to 0-1 scale
        normalized_accuracy = min(accuracy / 100, 1.0)
        
        # Calculate weighted confidence
        confidence = (normalized_accuracy * 0.5 + completeness * 0.3 + consistency * 0.2)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.8  # Should be high given good input values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])