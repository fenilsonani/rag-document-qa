"""
Unit tests for query_intelligence.py module.
Tests intelligent query enhancement, suggestions, and analysis functions.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.query_intelligence import QueryIntelligence
from src.config import Config
from langchain.schema import Document


@pytest.fixture
def query_intelligence():
    """Create QueryIntelligence instance for testing."""
    return QueryIntelligence()


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata={"source": "ai_guide.pdf", "page": 1}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers to process data.",
            metadata={"source": "ml_fundamentals.pdf", "page": 3}
        ),
        Document(
            page_content="Natural language processing enables computers to understand human language.",
            metadata={"source": "nlp_overview.pdf", "page": 2}
        )
    ]


@pytest.fixture
def sample_conversation_history():
    """Create sample conversation history for testing."""
    return [
        {
            "query": "What is machine learning?",
            "timestamp": datetime.now(),
            "response": "Machine learning is a branch of AI..."
        },
        {
            "query": "How does deep learning work?",
            "timestamp": datetime.now() - timedelta(minutes=5),
            "response": "Deep learning uses neural networks..."
        }
    ]


class TestQueryIntelligenceInit:
    """Test QueryIntelligence initialization."""
    
    def test_init_creates_correct_attributes(self):
        """Test that QueryIntelligence initializes with correct attributes."""
        qi = QueryIntelligence()
        
        assert hasattr(qi, 'config')
        assert hasattr(qi, 'query_history')
        assert hasattr(qi, 'document_themes')
        assert hasattr(qi, 'user_interests')
        assert hasattr(qi, 'question_templates')
        
        assert isinstance(qi.query_history, list)
        assert isinstance(qi.document_themes, dict)
        assert isinstance(qi.user_interests, defaultdict)
        assert isinstance(qi.question_templates, dict)
    
    def test_init_creates_question_templates(self):
        """Test that question templates are properly initialized."""
        qi = QueryIntelligence()
        
        expected_categories = ['analysis', 'comparison', 'explanation', 'application']
        for category in expected_categories:
            assert category in qi.question_templates
            assert isinstance(qi.question_templates[category], list)
            assert len(qi.question_templates[category]) > 0


class TestEnhanceQuery:
    """Test enhance_query method."""
    
    def test_enhance_query_basic(self, query_intelligence, sample_documents):
        """Test basic query enhancement functionality."""
        query = "What is machine learning?"
        
        result = query_intelligence.enhance_query(query, sample_documents)
        
        assert isinstance(result, dict)
        assert 'original_query' in result
        assert 'enhanced_query' in result
        assert 'query_analysis' in result
        assert 'suggestions' in result
        assert 'expansion_terms' in result
        
        assert result['original_query'] == query
        assert isinstance(result['suggestions'], list)
    
    def test_enhance_query_with_conversation_history(self, query_intelligence, sample_documents, sample_conversation_history):
        """Test query enhancement with conversation history."""
        query = "How does it relate to AI?"
        
        result = query_intelligence.enhance_query(
            query, 
            sample_documents, 
            conversation_history=sample_conversation_history
        )
        
        assert isinstance(result, dict)
        assert result['original_query'] == query
        assert 'enhanced_query' in result
    
    def test_enhance_query_empty_documents(self, query_intelligence):
        """Test query enhancement with empty document list."""
        query = "What is machine learning?"
        
        result = query_intelligence.enhance_query(query, [])
        
        assert isinstance(result, dict)
        assert result['original_query'] == query
        # Should still return suggestions even without documents
        assert 'suggestions' in result
    
    def test_enhance_query_complex_query(self, query_intelligence, sample_documents):
        """Test enhancement of complex multi-part query."""
        query = "Compare machine learning and deep learning approaches for natural language processing tasks"
        
        result = query_intelligence.enhance_query(query, sample_documents)
        
        assert isinstance(result, dict)
        assert result['original_query'] == query
        # Complex queries should have analysis
        assert 'query_analysis' in result
        assert result['query_analysis']['difficulty'] in ['easy', 'medium', 'hard']


class TestQueryAnalysis:
    """Test query analysis methods."""
    
    def test_analyze_query_intent_question(self, query_intelligence):
        """Test intent analysis for question queries."""
        query = "What is machine learning?"
        
        result = query_intelligence._analyze_query_intent(query)
        
        assert isinstance(result, dict)
        assert 'intent_type' in result
        assert 'confidence' in result
        assert 'keywords' in result
        assert result['intent_type'] in ['question', 'comparison', 'explanation', 'analysis']
        assert 0 <= result['confidence'] <= 1
    
    def test_analyze_query_intent_comparison(self, query_intelligence):
        """Test intent analysis for comparison queries."""
        query = "Compare machine learning vs deep learning"
        
        result = query_intelligence._analyze_query_intent(query)
        
        assert isinstance(result, dict)
        assert result['intent_type'] == 'comparison'
        assert 'keywords' in result
    
    def test_estimate_query_difficulty(self, query_intelligence):
        """Test query difficulty estimation."""
        easy_query = "What is AI?"
        medium_query = "How does machine learning classification work?"
        hard_query = "Explain the mathematical foundations of backpropagation in deep neural networks"
        
        assert query_intelligence._estimate_query_difficulty(easy_query) == 'easy'
        assert query_intelligence._estimate_query_difficulty(medium_query) in ['easy', 'medium']
        assert query_intelligence._estimate_query_difficulty(hard_query) in ['medium', 'hard']
    
    def test_extract_query_concepts(self, query_intelligence):
        """Test concept extraction from queries."""
        query = "Machine learning algorithms for natural language processing"
        
        concepts = query_intelligence._extract_query_concepts(query)
        
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        # Should extract meaningful concepts
        assert any('machine learning' in concept.lower() for concept in concepts)


class TestQueryExpansion:
    """Test query expansion functionality."""
    
    @patch('src.query_intelligence.TfidfVectorizer')
    def test_expand_query_terms(self, mock_tfidf, query_intelligence, sample_documents):
        """Test query term expansion using TF-IDF."""
        query = "machine learning"
        
        # Mock TF-IDF vectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = Mock()
        mock_vectorizer.transform.return_value = Mock()
        mock_vectorizer.get_feature_names_out.return_value = ['machine', 'learning', 'algorithm', 'neural']
        mock_tfidf.return_value = mock_vectorizer
        
        result = query_intelligence._expand_query_terms(query, sample_documents)
        
        assert isinstance(result, dict)
        assert 'expanded_terms' in result
        assert 'similarity_scores' in result
    
    def test_expand_query_terms_empty_documents(self, query_intelligence):
        """Test query expansion with empty documents."""
        query = "machine learning"
        
        result = query_intelligence._expand_query_terms(query, [])
        
        assert isinstance(result, dict)
        assert 'expanded_terms' in result
        # Should handle empty documents gracefully
        assert isinstance(result['expanded_terms'], list)


class TestSuggestionGeneration:
    """Test suggestion generation methods."""
    
    def test_generate_intelligent_suggestions(self, query_intelligence, sample_documents, sample_conversation_history):
        """Test intelligent suggestion generation."""
        query = "What is machine learning?"
        
        suggestions = query_intelligence._generate_intelligent_suggestions(
            query, sample_documents, sample_conversation_history
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check suggestion structure
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert 'question' in suggestion
            assert 'type' in suggestion
            assert 'relevance_score' in suggestion
            assert 0 <= suggestion['relevance_score'] <= 1
    
    def test_generate_document_specific_suggestions(self, query_intelligence, sample_documents):
        """Test document-specific suggestion generation."""
        query = "machine learning"
        
        suggestions = query_intelligence._generate_document_specific_suggestions(query, sample_documents)
        
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert 'question' in suggestion
            assert 'source_document' in suggestion
            assert 'relevance_score' in suggestion
    
    def test_generate_follow_up_questions(self, query_intelligence):
        """Test follow-up question generation."""
        query = "What is machine learning?"
        concepts = ["machine learning", "artificial intelligence", "algorithms"]
        document_themes = {"technology": ["AI", "ML"], "applications": ["automation"]}
        
        follow_ups = query_intelligence._generate_follow_up_questions(query, concepts, document_themes)
        
        assert isinstance(follow_ups, list)
        for question in follow_ups:
            assert isinstance(question, dict)
            assert 'question' in question
            assert 'type' in question
            assert 'concepts' in question


class TestDocumentAnalysis:
    """Test document analysis methods."""
    
    def test_extract_document_themes(self, query_intelligence, sample_documents):
        """Test document theme extraction."""
        themes = query_intelligence._extract_document_themes(sample_documents)
        
        assert isinstance(themes, dict)
        # Should extract themes from documents
        assert len(themes) > 0
        
        for theme, keywords in themes.items():
            assert isinstance(theme, str)
            assert isinstance(keywords, list)
    
    def test_extract_recent_topics(self, query_intelligence, sample_conversation_history):
        """Test recent topic extraction from conversation history."""
        topics = query_intelligence._extract_recent_topics(sample_conversation_history)
        
        assert isinstance(topics, list)
        # Should extract meaningful topics
        assert len(topics) >= 0


class TestComplexityAnalysis:
    """Test query complexity analysis."""
    
    def test_analyze_query_complexity(self, query_intelligence):
        """Test comprehensive query complexity analysis."""
        simple_query = "What is AI?"
        complex_query = "Analyze the impact of attention mechanisms in transformer architectures on sequence-to-sequence tasks"
        
        simple_result = query_intelligence._analyze_query_complexity(simple_query)
        complex_result = query_intelligence._analyze_query_complexity(complex_query)
        
        assert isinstance(simple_result, dict)
        assert isinstance(complex_result, dict)
        
        # Check required fields
        for result in [simple_result, complex_result]:
            assert 'complexity_level' in result
            assert 'factors' in result
            assert 'word_count' in result
            assert 'technical_terms' in result
        
        # Complex query should have higher complexity indicators
        assert complex_result['word_count'] > simple_result['word_count']
    
    def test_get_complexity_suggestions(self, query_intelligence):
        """Test complexity-based suggestion generation."""
        suggestions_easy = query_intelligence._get_complexity_suggestions('easy', "What is AI?")
        suggestions_hard = query_intelligence._get_complexity_suggestions('hard', "Complex technical query")
        
        assert isinstance(suggestions_easy, list)
        assert isinstance(suggestions_hard, list)
        
        # Should provide different suggestions based on complexity
        assert len(suggestions_easy) > 0
        assert len(suggestions_hard) > 0


class TestQueryAlternatives:
    """Test query alternative generation."""
    
    def test_generate_query_alternatives(self, query_intelligence):
        """Test generation of alternative query phrasings."""
        query = "What is machine learning?"
        concepts = ["machine learning", "AI", "algorithms"]
        
        alternatives = query_intelligence._generate_query_alternatives(query, concepts)
        
        assert isinstance(alternatives, list)
        for alt in alternatives:
            assert isinstance(alt, dict)
            assert 'alternative' in alt
            assert 'type' in alt
            assert alt['type'] in ['simpler', 'more_specific', 'broader']


class TestTrendingQueries:
    """Test trending query functionality."""
    
    def test_get_trending_queries_empty_history(self, query_intelligence):
        """Test trending queries with empty history."""
        trending = query_intelligence.get_trending_queries()
        
        assert isinstance(trending, list)
        # Empty history should return empty list
        assert len(trending) == 0
    
    def test_get_trending_queries_with_history(self, query_intelligence):
        """Test trending queries with query history."""
        # Add some query history
        query_intelligence.query_history = [
            {'query': 'machine learning', 'timestamp': datetime.now()},
            {'query': 'deep learning', 'timestamp': datetime.now() - timedelta(hours=1)},
            {'query': 'machine learning', 'timestamp': datetime.now() - timedelta(hours=2)},
        ]
        
        trending = query_intelligence.get_trending_queries(time_window_hours=24)
        
        assert isinstance(trending, list)
        if trending:  # If there are trending queries
            for trend in trending:
                assert isinstance(trend, dict)
                assert 'query' in trend
                assert 'frequency' in trend


class TestExplorationPaths:
    """Test exploration path suggestions."""
    
    def test_suggest_exploration_paths(self, query_intelligence, sample_documents):
        """Test exploration path generation."""
        paths = query_intelligence.suggest_exploration_paths(sample_documents)
        
        assert isinstance(paths, list)
        for path in paths:
            assert isinstance(path, dict)
            assert 'theme' in path
            assert 'questions' in path
            assert 'difficulty' in path
            assert isinstance(path['questions'], list)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_enhance_query_empty_string(self, query_intelligence, sample_documents):
        """Test enhancement of empty query."""
        result = query_intelligence.enhance_query("", sample_documents)
        
        assert isinstance(result, dict)
        assert result['original_query'] == ""
        # Should handle empty query gracefully
    
    def test_enhance_query_very_long_query(self, query_intelligence, sample_documents):
        """Test enhancement of very long query."""
        long_query = " ".join(["word"] * 1000)
        
        result = query_intelligence.enhance_query(long_query, sample_documents)
        
        assert isinstance(result, dict)
        assert result['original_query'] == long_query
    
    def test_analyze_query_intent_special_characters(self, query_intelligence):
        """Test intent analysis with special characters."""
        query = "What is ML? @#$%^&*()"
        
        result = query_intelligence._analyze_query_intent(query)
        
        assert isinstance(result, dict)
        # Should handle special characters without crashing
        assert 'intent_type' in result
    
    def test_extract_concepts_unicode(self, query_intelligence):
        """Test concept extraction with unicode characters."""
        query = "Машинное обучение and 機械学習"
        
        concepts = query_intelligence._extract_query_concepts(query)
        
        assert isinstance(concepts, list)
        # Should handle unicode gracefully


class TestQueryHistoryUpdate:
    """Test query history update functionality."""
    
    def test_update_query_history(self, query_intelligence):
        """Test query history update."""
        initial_length = len(query_intelligence.query_history)
        query = "test query"
        
        query_intelligence._update_query_history(query)
        
        assert len(query_intelligence.query_history) == initial_length + 1
        latest_entry = query_intelligence.query_history[-1]
        assert latest_entry['query'] == query
        assert 'timestamp' in latest_entry
        assert isinstance(latest_entry['timestamp'], datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])