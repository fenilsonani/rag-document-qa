"""
Advanced Document Intelligence System - Unique RAG Enhancement
Provides intelligent document analysis, summarization, and insights extraction.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

from langchain.schema import Document
from .config import Config


class DocumentIntelligence:
    """Advanced document analysis and intelligence extraction."""
    
    def __init__(self):
        self.config = Config()
        self._ensure_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
    
    def _ensure_nltk_data(self):
        """Download required NLTK data."""
        required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    pass  # Continue if download fails
    
    def generate_document_insights(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate comprehensive document insights and analysis."""
        if not documents:
            return {"error": "No documents provided"}
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        full_text = "\n".join(texts)
        
        insights = {
            "summary": self._generate_intelligent_summary(texts),
            "key_concepts": self._extract_key_concepts(full_text),
            "named_entities": self._extract_named_entities(full_text),
            "document_statistics": self._calculate_document_stats(documents),
            "complexity_analysis": self._analyze_complexity(full_text),
            "topic_clusters": self._perform_topic_clustering(texts),
            "readability_score": self._calculate_readability(full_text),
            "temporal_analysis": self._extract_temporal_info(full_text),
            "quality_assessment": self._assess_document_quality(documents),
            "generated_at": datetime.now().isoformat()
        }
        
        return insights
    
    def _generate_intelligent_summary(self, texts: List[str]) -> Dict[str, Any]:
        """Generate multi-level intelligent summaries."""
        full_text = "\n".join(texts)
        sentences = sent_tokenize(full_text)
        
        if len(sentences) < 3:
            return {
                "executive_summary": full_text[:500] + "..." if len(full_text) > 500 else full_text,
                "key_points": sentences,
                "word_count": len(full_text.split())
            }
        
        # Score sentences by importance
        scored_sentences = self._score_sentences(sentences)
        
        # Create different summary levels
        executive_summary = self._create_executive_summary(scored_sentences[:3])
        key_points = [sent for sent, score in scored_sentences[:5]]
        
        return {
            "executive_summary": executive_summary,
            "key_points": key_points,
            "total_sentences": len(sentences),
            "word_count": len(full_text.split()),
            "confidence_score": min(0.95, len(sentences) / 20)  # Higher confidence with more content
        }
    
    def _score_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        """Score sentences by importance using multiple factors."""
        if not sentences:
            return []
        
        # TF-IDF scoring
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            sentence_scores = np.mean(tfidf_matrix.toarray(), axis=1)
        except:
            sentence_scores = [1.0] * len(sentences)
        
        # Position scoring (first and last sentences are more important)
        position_scores = []
        for i, _ in enumerate(sentences):
            if i == 0 or i == len(sentences) - 1:
                position_scores.append(1.2)
            elif i < len(sentences) * 0.3:
                position_scores.append(1.1)
            else:
                position_scores.append(1.0)
        
        # Length scoring (prefer medium-length sentences)
        length_scores = []
        for sent in sentences:
            word_count = len(sent.split())
            if 10 <= word_count <= 25:
                length_scores.append(1.1)
            elif word_count < 5:
                length_scores.append(0.8)
            else:
                length_scores.append(1.0)
        
        # Combine scores
        final_scores = []
        for i, sent in enumerate(sentences):
            combined_score = (
                sentence_scores[i] * 0.6 +
                position_scores[i] * 0.2 +
                length_scores[i] * 0.2
            )
            final_scores.append((sent, combined_score))
        
        return sorted(final_scores, key=lambda x: x[1], reverse=True)
    
    def _create_executive_summary(self, top_sentences: List[Tuple[str, float]]) -> str:
        """Create a coherent executive summary from top sentences."""
        sentences = [sent for sent, score in top_sentences]
        return " ".join(sentences)
    
    def _extract_key_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract and rank key concepts from the text."""
        # Tokenize and clean
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 3]
        
        # Get POS tags
        try:
            pos_tags = pos_tag(words)
            # Focus on nouns and adjectives
            important_words = [word for word, pos in pos_tags if pos.startswith(('NN', 'JJ', 'VB'))]
        except:
            important_words = words
        
        # Count frequency
        word_freq = Counter(important_words)
        
        # Extract n-grams for compound concepts
        text_words = text.lower().split()
        bigrams = [f"{text_words[i]} {text_words[i+1]}" for i in range(len(text_words)-1)]
        trigrams = [f"{text_words[i]} {text_words[i+1]} {text_words[i+2]}" for i in range(len(text_words)-2)]
        
        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)
        
        # Combine and rank concepts
        concepts = []
        
        # Single words
        for word, freq in word_freq.most_common(10):
            concepts.append({
                "concept": word,
                "frequency": freq,
                "type": "single_word",
                "importance": freq / len(important_words)
            })
        
        # Bigrams
        for bigram, freq in bigram_freq.most_common(5):
            if freq > 2:  # Only include if appears multiple times
                concepts.append({
                    "concept": bigram,
                    "frequency": freq,
                    "type": "compound",
                    "importance": freq / len(bigrams)
                })
        
        # Trigrams
        for trigram, freq in trigram_freq.most_common(3):
            if freq > 1:
                concepts.append({
                    "concept": trigram,
                    "frequency": freq,
                    "type": "phrase",
                    "importance": freq / len(trigrams)
                })
        
        return sorted(concepts, key=lambda x: x['importance'], reverse=True)[:15]
    
    def _extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using NLTK."""
        entities = {
            "PERSON": [],
            "ORGANIZATION": [],
            "LOCATION": [],
            "DATE": [],
            "MONEY": [],
            "OTHER": []
        }
        
        try:
            # Tokenize and tag
            sentences = sent_tokenize(text)
            for sentence in sentences[:10]:  # Limit for performance
                words = word_tokenize(sentence)
                pos_tags = pos_tag(words)
                chunks = ne_chunk(pos_tags)
                
                current_entity = []
                current_label = None
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        if current_entity and current_label:
                            entity_text = " ".join(current_entity)
                            if current_label in entities:
                                entities[current_label].append(entity_text)
                            else:
                                entities["OTHER"].append(entity_text)
                        
                        current_entity = [leaf[0] for leaf in chunk.leaves()]
                        current_label = chunk.label()
                    else:
                        if current_entity and current_label:
                            entity_text = " ".join(current_entity)
                            if current_label in entities:
                                entities[current_label].append(entity_text)
                            else:
                                entities["OTHER"].append(entity_text)
                            current_entity = []
                            current_label = None
                
                # Handle last entity
                if current_entity and current_label:
                    entity_text = " ".join(current_entity)
                    if current_label in entities:
                        entities[current_label].append(entity_text)
                    else:
                        entities["OTHER"].append(entity_text)
        
        except Exception as e:
            # Fallback: simple pattern matching
            entities["DATE"] = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text)
            entities["MONEY"] = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        
        # Remove duplicates and empty entries
        for key in entities:
            entities[key] = list(set([e for e in entities[key] if e.strip()]))
        
        return entities
    
    def _calculate_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Calculate comprehensive document statistics."""
        full_text = "\n".join([doc.page_content for doc in documents])
        
        stats = {
            "total_documents": len(documents),
            "total_characters": len(full_text),
            "total_words": len(full_text.split()),
            "total_sentences": len(sent_tokenize(full_text)),
            "average_words_per_sentence": 0,
            "unique_words": 0,
            "vocabulary_richness": 0,
            "document_types": {},
            "average_document_length": 0
        }
        
        words = full_text.lower().split()
        unique_words = set(words)
        
        stats["unique_words"] = len(unique_words)
        stats["vocabulary_richness"] = len(unique_words) / len(words) if words else 0
        stats["average_words_per_sentence"] = len(words) / stats["total_sentences"] if stats["total_sentences"] > 0 else 0
        stats["average_document_length"] = len(words) / len(documents) if documents else 0
        
        # Document types
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            stats["document_types"][file_type] = stats["document_types"].get(file_type, 0) + 1
        
        return stats
    
    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity using multiple metrics."""
        words = text.split()
        sentences = sent_tokenize(text)
        
        if not words or not sentences:
            return {"error": "Insufficient text for analysis"}
        
        # Basic metrics
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = sum([self._count_syllables(word) for word in words]) / len(words)
        
        # Flesch Reading Ease Score
        flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        
        # Complexity classification
        if flesch_score >= 90:
            complexity_level = "Very Easy"
        elif flesch_score >= 80:
            complexity_level = "Easy"
        elif flesch_score >= 70:
            complexity_level = "Fairly Easy"
        elif flesch_score >= 60:
            complexity_level = "Standard"
        elif flesch_score >= 50:
            complexity_level = "Fairly Difficult"
        elif flesch_score >= 30:
            complexity_level = "Difficult"
        else:
            complexity_level = "Very Difficult"
        
        return {
            "flesch_score": round(flesch_score, 1),
            "complexity_level": complexity_level,
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "recommended_audience": self._get_audience_recommendation(flesch_score)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _get_audience_recommendation(self, flesch_score: float) -> str:
        """Get audience recommendation based on Flesch score."""
        if flesch_score >= 90:
            return "5th grade students"
        elif flesch_score >= 80:
            return "6th grade students"
        elif flesch_score >= 70:
            return "7th grade students"
        elif flesch_score >= 60:
            return "8th-9th grade students"
        elif flesch_score >= 50:
            return "High school students"
        elif flesch_score >= 30:
            return "College students"
        else:
            return "Graduate students/professionals"
    
    def _perform_topic_clustering(self, texts: List[str]) -> Dict[str, Any]:
        """Perform topic clustering on document chunks."""
        if len(texts) < 3:
            return {"error": "Insufficient documents for clustering"}
        
        try:
            # Vectorize texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Determine optimal number of clusters
            n_clusters = min(5, max(2, len(texts) // 3))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Get top terms for each cluster
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            clusters = {}
            
            for i in range(n_clusters):
                # Get top terms for this cluster
                center = kmeans.cluster_centers_[i]
                top_indices = center.argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                
                # Get documents in this cluster
                cluster_docs = [j for j, label in enumerate(cluster_labels) if label == i]
                
                clusters[f"Cluster_{i+1}"] = {
                    "top_terms": top_terms,
                    "document_count": len(cluster_docs),
                    "representative_snippet": texts[cluster_docs[0]][:200] + "..." if cluster_docs else ""
                }
            
            return {
                "clusters": clusters,
                "total_clusters": n_clusters,
                "clustering_quality": "Good" if n_clusters >= 2 else "Basic"
            }
        
        except Exception as e:
            return {"error": f"Clustering failed: {str(e)}"}
    
    def _calculate_readability(self, text: str) -> Dict[str, Any]:
        """Calculate various readability metrics."""
        words = text.split()
        sentences = sent_tokenize(text)
        
        if not words or not sentences:
            return {"error": "Insufficient text"}
        
        # Basic metrics
        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum([self._count_syllables(word) for word in words])
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        
        # Automated Readability Index
        chars = sum(len(word) for word in words)
        ari = 4.71 * (chars / total_words) + 0.5 * (total_words / total_sentences) - 21.43
        
        return {
            "flesch_kincaid_grade": round(max(0, fk_grade), 1),
            "automated_readability_index": round(max(0, ari), 1),
            "estimated_reading_time_minutes": round(total_words / 200, 1),  # 200 WPM average
            "complexity_indicators": {
                "long_sentences": sum(1 for s in sentences if len(s.split()) > 20),
                "complex_words": sum(1 for w in words if self._count_syllables(w) > 2),
                "passive_voice_indicators": len(re.findall(r'\b(was|were|been|being)\s+\w+ed\b', text, re.IGNORECASE))
            }
        }
    
    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information and trends from text."""
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
            r'\b\d{4}\b',                           # Year only
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        dates_found = []
        for pattern in date_patterns:
            dates_found.extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Time-related keywords
        temporal_keywords = {
            "past": re.findall(r'\b(previously|formerly|historically|past|ago|before|earlier|prior)\b', text, re.IGNORECASE),
            "present": re.findall(r'\b(currently|now|today|present|modern|contemporary)\b', text, re.IGNORECASE),
            "future": re.findall(r'\b(will|future|upcoming|planned|projected|anticipated|expected)\b', text, re.IGNORECASE)
        }
        
        return {
            "dates_mentioned": list(set(dates_found)),
            "temporal_focus": {
                "past_references": len(temporal_keywords["past"]),
                "present_references": len(temporal_keywords["present"]),
                "future_references": len(temporal_keywords["future"])
            },
            "dominant_timeframe": max(temporal_keywords.keys(), key=lambda k: len(temporal_keywords[k])) if any(temporal_keywords.values()) else "neutral"
        }
    
    def _assess_document_quality(self, documents: List[Document]) -> Dict[str, Any]:
        """Assess overall document quality and reliability."""
        if not documents:
            return {"error": "No documents to assess"}
        
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # Quality indicators
        quality_score = 100  # Start with perfect score
        issues = []
        
        # Check for minimal content
        word_count = len(full_text.split())
        if word_count < 100:
            quality_score -= 20
            issues.append("Very short content")
        
        # Check for repetitive content
        sentences = sent_tokenize(full_text)
        unique_sentences = set(sentences)
        if len(unique_sentences) / len(sentences) < 0.8:
            quality_score -= 15
            issues.append("Repetitive content detected")
        
        # Check for proper sentence structure
        malformed_sentences = sum(1 for s in sentences if len(s.split()) < 3 or not s.strip().endswith(('.', '!', '?')))
        if malformed_sentences / len(sentences) > 0.1:
            quality_score -= 10
            issues.append("Formatting issues detected")
        
        # Check for metadata completeness
        metadata_completeness = 0
        for doc in documents:
            if doc.metadata.get('filename'):
                metadata_completeness += 1
            if doc.metadata.get('source'):
                metadata_completeness += 1
        
        metadata_score = (metadata_completeness / (len(documents) * 2)) * 100
        
        # Final quality assessment
        quality_score = max(0, quality_score)
        
        if quality_score >= 90:
            quality_level = "Excellent"
        elif quality_score >= 80:
            quality_level = "Good"
        elif quality_score >= 70:
            quality_level = "Fair"
        elif quality_score >= 60:
            quality_level = "Poor"
        else:
            quality_level = "Very Poor"
        
        return {
            "overall_score": quality_score,
            "quality_level": quality_level,
            "metadata_completeness": round(metadata_score, 1),
            "content_diversity": round((len(unique_sentences) / len(sentences)) * 100, 1),
            "issues_detected": issues,
            "recommendations": self._generate_quality_recommendations(quality_score, issues)
        }
    
    def _generate_quality_recommendations(self, score: float, issues: List[str]) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        if score < 70:
            recommendations.append("Consider adding more detailed content")
        
        if "Repetitive content detected" in issues:
            recommendations.append("Remove duplicate or repetitive sections")
        
        if "Formatting issues detected" in issues:
            recommendations.append("Improve document formatting and structure")
        
        if "Very short content" in issues:
            recommendations.append("Expand content with more detailed information")
        
        if not recommendations:
            recommendations.append("Document quality is good - no major improvements needed")
        
        return recommendations