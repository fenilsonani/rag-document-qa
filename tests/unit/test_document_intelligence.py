"""
Unit tests for document_intelligence.py module.
Tests document analysis, insights generation, and quality assessment functions.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock NLTK before importing document_intelligence
with patch.dict('sys.modules', {'nltk': MagicMock(), 'nltk.corpus': MagicMock(), 'nltk.tokenize': MagicMock()}):
    from src.document_intelligence import DocumentIntelligence
    from langchain.schema import Document


@pytest.fixture
def document_intelligence():
    """Create DocumentIntelligence instance for testing."""
    with patch.object(DocumentIntelligence, '_ensure_nltk_data'):
        return DocumentIntelligence()


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="""Machine learning is a method of data analysis that automates analytical model building. 
            It is a branch of artificial intelligence based on the idea that systems can learn from data, 
            identify patterns and make decisions with minimal human intervention. Machine learning algorithms 
            build a mathematical model based on training data in order to make predictions or decisions 
            without being explicitly programmed to perform the task.""",
            metadata={"source": "ml_intro.pdf", "page": 1}
        ),
        Document(
            page_content="""Deep learning is part of a broader family of machine learning methods based on 
            artificial neural networks with representation learning. Learning can be supervised, 
            semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, 
            deep belief networks, recurrent neural networks and convolutional neural networks have been 
            applied to fields including computer vision, speech recognition, natural language processing.""",
            metadata={"source": "deep_learning.pdf", "page": 2}
        ),
        Document(
            page_content="""Natural language processing is a subfield of linguistics, computer science, 
            and artificial intelligence concerned with the interactions between computers and human language, 
            in particular how to program computers to process and analyze large amounts of natural language data.""",
            metadata={"source": "nlp_guide.pdf", "page": 1}
        )
    ]


@pytest.fixture 
def short_documents():
    """Create short documents for testing edge cases."""
    return [
        Document(
            page_content="Short text.",
            metadata={"source": "short.pdf", "page": 1}
        ),
        Document(
            page_content="AI is cool.",
            metadata={"source": "brief.pdf", "page": 1}
        )
    ]


class TestDocumentIntelligenceInit:
    """Test DocumentIntelligence initialization."""
    
    @patch.object(DocumentIntelligence, '_ensure_nltk_data')
    def test_initialization(self, mock_ensure_nltk):
        """Test DocumentIntelligence initialization."""
        di = DocumentIntelligence()
        
        assert hasattr(di, 'config')
        assert hasattr(di, 'stop_words')
        mock_ensure_nltk.assert_called_once()
    
    @patch('nltk.download')
    @patch('nltk.data.find', side_effect=LookupError())
    def test_ensure_nltk_data_download(self, mock_find, mock_download):
        """Test NLTK data downloading."""
        with patch.object(DocumentIntelligence, '_ensure_nltk_data', DocumentIntelligence._ensure_nltk_data):
            di = DocumentIntelligence()
            
            # Should attempt to download required NLTK data
            assert mock_download.call_count > 0


class TestGenerateDocumentInsights:
    """Test generate_document_insights method."""
    
    @patch('src.document_intelligence.DocumentIntelligence._generate_intelligent_summary')
    @patch('src.document_intelligence.DocumentIntelligence._extract_key_concepts')
    @patch('src.document_intelligence.DocumentIntelligence._calculate_document_stats')
    def test_generate_document_insights_basic(self, mock_stats, mock_concepts, mock_summary, 
                                            document_intelligence, sample_documents):
        """Test basic document insights generation."""
        # Setup mocks
        mock_summary.return_value = {"summary": "Test summary", "key_points": []}
        mock_concepts.return_value = [{"concept": "machine learning", "score": 0.9}]
        mock_stats.return_value = {"total_documents": 3, "total_words": 100}
        
        insights = document_intelligence.generate_document_insights(sample_documents)
        
        assert isinstance(insights, dict)
        assert 'summary' in insights
        assert 'key_concepts' in insights
        assert 'statistics' in insights
        assert 'analysis_metadata' in insights
        
        # Verify method calls
        mock_summary.assert_called_once()
        mock_concepts.assert_called_once()
        mock_stats.assert_called_once_with(sample_documents)
    
    def test_generate_document_insights_empty_documents(self, document_intelligence):
        """Test insights generation with empty document list."""
        insights = document_intelligence.generate_document_insights([])
        
        assert isinstance(insights, dict)
        # Should handle empty documents gracefully
        assert 'error' not in insights or not insights['error']
    
    def test_generate_document_insights_single_document(self, document_intelligence, sample_documents):
        """Test insights generation with single document."""
        single_doc = [sample_documents[0]]
        
        insights = document_intelligence.generate_document_insights(single_doc)
        
        assert isinstance(insights, dict)
        assert 'summary' in insights
        assert 'statistics' in insights


class TestIntelligentSummary:
    """Test intelligent summary generation."""
    
    @patch('src.document_intelligence.TfidfVectorizer')
    def test_generate_intelligent_summary(self, mock_tfidf, document_intelligence):
        """Test intelligent summary generation."""
        texts = [
            "Machine learning is a method of data analysis.",
            "Deep learning uses neural networks.",
            "Natural language processing enables computers to understand language."
        ]
        
        # Mock TF-IDF
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = Mock()
        mock_vectorizer.get_feature_names_out.return_value = ['machine', 'learning', 'data']
        mock_tfidf.return_value = mock_vectorizer
        
        with patch.object(document_intelligence, '_score_sentences') as mock_score:
            mock_score.return_value = [
                ("Machine learning is important.", 0.9),
                ("Data analysis is key.", 0.8)
            ]
            
            summary = document_intelligence._generate_intelligent_summary(texts)
            
            assert isinstance(summary, dict)
            assert 'executive_summary' in summary
            assert 'key_points' in summary
            assert 'confidence_score' in summary
    
    def test_generate_intelligent_summary_short_texts(self, document_intelligence):
        """Test summary generation with very short texts."""
        short_texts = ["AI.", "ML.", "NLP."]
        
        summary = document_intelligence._generate_intelligent_summary(short_texts)
        
        assert isinstance(summary, dict)
        # Should handle short texts gracefully


class TestSentenceScoring:
    """Test sentence scoring functionality."""
    
    @patch('src.document_intelligence.TfidfVectorizer')
    def test_score_sentences(self, mock_tfidf, document_intelligence):
        """Test sentence scoring mechanism."""
        sentences = [
            "Machine learning is a powerful tool for data analysis.",
            "It can identify patterns in large datasets.",
            "This capability makes it valuable for businesses."
        ]
        
        # Mock TF-IDF vectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
        mock_tfidf.return_value = mock_vectorizer
        
        scores = document_intelligence._score_sentences(sentences)
        
        assert isinstance(scores, list)
        assert len(scores) == len(sentences)
        
        for sentence, score in scores:
            assert isinstance(sentence, str)
            assert isinstance(score, (int, float))
            assert score >= 0


class TestKeyConceptExtraction:
    """Test key concept extraction."""
    
    @patch('src.document_intelligence.TfidfVectorizer')
    def test_extract_key_concepts(self, mock_tfidf, document_intelligence):
        """Test key concept extraction from text."""
        text = """Machine learning algorithms can automatically learn and improve from experience 
                 without being explicitly programmed. Deep learning is a subset of machine learning 
                 that uses neural networks with multiple layers."""
        
        # Mock TF-IDF
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = Mock()
        mock_vectorizer.get_feature_names_out.return_value = ['machine', 'learning', 'deep', 'neural']
        mock_vectorizer.transform.return_value = [[0.8, 0.7, 0.5, 0.3]]
        mock_tfidf.return_value = mock_vectorizer
        
        concepts = document_intelligence._extract_key_concepts(text)
        
        assert isinstance(concepts, list)
        for concept in concepts:
            assert isinstance(concept, dict)
            assert 'concept' in concept
            assert 'importance_score' in concept
            assert 'category' in concept
    
    def test_extract_key_concepts_empty_text(self, document_intelligence):
        """Test concept extraction from empty text."""
        concepts = document_intelligence._extract_key_concepts("")
        
        assert isinstance(concepts, list)
        assert len(concepts) == 0


class TestNamedEntityExtraction:
    """Test named entity extraction."""
    
    @patch('nltk.ne_chunk')
    @patch('nltk.pos_tag')
    @patch('nltk.word_tokenize')
    def test_extract_named_entities(self, mock_tokenize, mock_pos_tag, mock_ne_chunk, document_intelligence):
        """Test named entity extraction."""
        text = "John works at Google in California."
        
        # Mock NLTK functions
        mock_tokenize.return_value = ['John', 'works', 'at', 'Google', 'in', 'California', '.']
        mock_pos_tag.return_value = [('John', 'NNP'), ('works', 'VBZ'), ('at', 'IN'), 
                                     ('Google', 'NNP'), ('in', 'IN'), ('California', 'NNP'), ('.', '.')]
        
        # Mock named entity tree
        mock_tree = Mock()
        mock_tree.__iter__ = Mock(return_value=iter([
            Mock(label=Mock(return_value='PERSON'), leaves=Mock(return_value=[('John', 'NNP')])),
            Mock(label=Mock(return_value='ORGANIZATION'), leaves=Mock(return_value=[('Google', 'NNP')])),
            Mock(label=Mock(return_value='GPE'), leaves=Mock(return_value=[('California', 'NNP')]))
        ]))
        mock_ne_chunk.return_value = mock_tree
        
        entities = document_intelligence._extract_named_entities(text)
        
        assert isinstance(entities, dict)
        # Should contain different entity types
        expected_types = ['PERSON', 'ORGANIZATION', 'GPE']
        for entity_type in expected_types:
            if entity_type in entities:
                assert isinstance(entities[entity_type], list)


class TestDocumentStatistics:
    """Test document statistics calculation."""
    
    def test_calculate_document_stats(self, document_intelligence, sample_documents):
        """Test document statistics calculation."""
        stats = document_intelligence._calculate_document_stats(sample_documents)
        
        assert isinstance(stats, dict)
        expected_keys = [
            'total_documents', 'total_words', 'total_sentences', 
            'avg_words_per_document', 'avg_sentences_per_document',
            'document_sources', 'readability_analysis'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_documents'] == len(sample_documents)
        assert stats['total_words'] > 0
        assert stats['total_sentences'] > 0
    
    def test_calculate_document_stats_empty_documents(self, document_intelligence):
        """Test statistics calculation with empty documents."""
        stats = document_intelligence._calculate_document_stats([])
        
        assert isinstance(stats, dict)
        assert stats['total_documents'] == 0
        assert stats['total_words'] == 0


class TestComplexityAnalysis:
    """Test text complexity analysis."""
    
    @patch('nltk.word_tokenize')
    @patch('nltk.sent_tokenize')
    def test_analyze_complexity(self, mock_sent_tokenize, mock_word_tokenize, document_intelligence):
        """Test text complexity analysis."""
        text = "This is a simple sentence. This is another sentence with more complexity."
        
        # Mock NLTK tokenizers
        mock_sent_tokenize.return_value = ["This is a simple sentence.", "This is another sentence with more complexity."]
        mock_word_tokenize.return_value = ["This", "is", "a", "simple", "sentence", "This", "is", "another", 
                                          "sentence", "with", "more", "complexity"]
        
        complexity = document_intelligence._analyze_complexity(text)
        
        assert isinstance(complexity, dict)
        expected_keys = ['avg_sentence_length', 'avg_word_length', 'complexity_score', 'difficulty_level']
        
        for key in expected_keys:
            assert key in complexity
        
        assert complexity['avg_sentence_length'] > 0
        assert complexity['avg_word_length'] > 0
        assert complexity['difficulty_level'] in ['easy', 'medium', 'hard']


class TestReadabilityAnalysis:
    """Test readability analysis."""
    
    @patch('nltk.word_tokenize')
    @patch('nltk.sent_tokenize')
    def test_calculate_readability(self, mock_sent_tokenize, mock_word_tokenize, document_intelligence):
        """Test readability score calculation."""
        text = "The cat sat on the mat. It was a sunny day."
        
        # Mock NLTK tokenizers
        mock_sent_tokenize.return_value = ["The cat sat on the mat.", "It was a sunny day."]
        mock_word_tokenize.return_value = ["The", "cat", "sat", "on", "the", "mat", "It", "was", "a", "sunny", "day"]
        
        readability = document_intelligence._calculate_readability(text)
        
        assert isinstance(readability, dict)
        expected_keys = ['flesch_reading_ease', 'flesch_kincaid_grade', 'audience']
        
        for key in expected_keys:
            assert key in readability
        
        assert 0 <= readability['flesch_reading_ease'] <= 100
        assert readability['flesch_kincaid_grade'] >= 0
    
    def test_count_syllables(self, document_intelligence):
        """Test syllable counting."""
        # Test common words
        assert document_intelligence._count_syllables("cat") == 1
        assert document_intelligence._count_syllables("computer") == 3
        assert document_intelligence._count_syllables("beautiful") == 3
        
        # Test edge cases
        assert document_intelligence._count_syllables("") == 0
        assert document_intelligence._count_syllables("a") == 1
    
    def test_get_audience_recommendation(self, document_intelligence):
        """Test audience recommendation based on readability."""
        # Very easy text
        audience = document_intelligence._get_audience_recommendation(90)
        assert "5th grade" in audience or "elementary" in audience.lower()
        
        # College level text
        audience = document_intelligence._get_audience_recommendation(30)
        assert "college" in audience.lower() or "graduate" in audience.lower()


class TestTopicClustering:
    """Test topic clustering functionality."""
    
    @patch('src.document_intelligence.TfidfVectorizer')
    @patch('src.document_intelligence.KMeans')
    def test_perform_topic_clustering(self, mock_kmeans, mock_tfidf, document_intelligence):
        """Test topic clustering analysis."""
        texts = [
            "Machine learning and artificial intelligence",
            "Deep learning neural networks",
            "Natural language processing"
        ]
        
        # Mock clustering components
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mock_vectorizer.get_feature_names_out.return_value = ['machine', 'deep', 'natural']
        mock_tfidf.return_value = mock_vectorizer
        
        mock_clustering = Mock()
        mock_clustering.fit_predict.return_value = [0, 1, 2]
        mock_clustering.cluster_centers_ = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mock_kmeans.return_value = mock_clustering
        
        clusters = document_intelligence._perform_topic_clustering(texts)
        
        assert isinstance(clusters, dict)
        assert 'topics' in clusters
        assert 'cluster_assignments' in clusters


class TestTemporalAnalysis:
    """Test temporal information extraction."""
    
    def test_extract_temporal_info(self, document_intelligence):
        """Test temporal information extraction."""
        text = "In 2020, the company grew by 25%. Last year was successful. Next month we launch."
        
        temporal_info = document_intelligence._extract_temporal_info(text)
        
        assert isinstance(temporal_info, dict)
        expected_keys = ['dates_mentioned', 'time_periods', 'temporal_expressions']
        
        for key in expected_keys:
            assert key in temporal_info


class TestDocumentQualityAssessment:
    """Test document quality assessment."""
    
    def test_assess_document_quality(self, document_intelligence, sample_documents):
        """Test document quality assessment."""
        assessment = document_intelligence._assess_document_quality(sample_documents)
        
        assert isinstance(assessment, dict)
        expected_keys = ['overall_score', 'quality_factors', 'recommendations']
        
        for key in expected_keys:
            assert key in assessment
        
        assert 0 <= assessment['overall_score'] <= 100
        assert isinstance(assessment['quality_factors'], dict)
        assert isinstance(assessment['recommendations'], list)
    
    def test_assess_document_quality_poor_documents(self, document_intelligence, short_documents):
        """Test quality assessment with poor quality documents."""
        assessment = document_intelligence._assess_document_quality(short_documents)
        
        assert isinstance(assessment, dict)
        # Short documents should have lower quality scores
        assert assessment['overall_score'] < 80
    
    def test_generate_quality_recommendations(self, document_intelligence):
        """Test quality recommendation generation."""
        # Test low score
        recommendations = document_intelligence._generate_quality_recommendations(30, ["short_content", "poor_readability"])
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Test high score
        recommendations = document_intelligence._generate_quality_recommendations(90, [])
        
        assert isinstance(recommendations, list)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_analyze_empty_text(self, document_intelligence):
        """Test analysis of empty text."""
        # Should handle empty text gracefully
        complexity = document_intelligence._analyze_complexity("")
        assert isinstance(complexity, dict)
        
        concepts = document_intelligence._extract_key_concepts("")
        assert isinstance(concepts, list)
        
        entities = document_intelligence._extract_named_entities("")
        assert isinstance(entities, dict)
    
    def test_analyze_very_long_text(self, document_intelligence):
        """Test analysis of very long text."""
        long_text = "This is a sentence. " * 1000
        
        complexity = document_intelligence._analyze_complexity(long_text)
        assert isinstance(complexity, dict)
        
        # Should handle long text without crashing
        concepts = document_intelligence._extract_key_concepts(long_text)
        assert isinstance(concepts, list)
    
    def test_unicode_text_handling(self, document_intelligence):
        """Test handling of unicode text."""
        unicode_text = "Machine learning について。人工知能 is important. Машинное обучение."
        
        # Should handle unicode gracefully
        complexity = document_intelligence._analyze_complexity(unicode_text)
        assert isinstance(complexity, dict)
        
        concepts = document_intelligence._extract_key_concepts(unicode_text)
        assert isinstance(concepts, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])