"""
Intelligent Query Enhancement and Question Suggestion Engine
Advanced system for query expansion, intelligent suggestions, and adaptive questioning.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

from langchain.schema import Document
from .config import Config


class QueryIntelligence:
    """Advanced query enhancement and intelligent question suggestion system."""
    
    def __init__(self):
        self.config = Config()
        self.query_history = []
        self.document_themes = {}
        self.user_interests = defaultdict(int)
        self.question_templates = {
            "analysis": [
                "What are the main {concept} in {domain}?",
                "How does {concept1} relate to {concept2}?",
                "What are the key differences between {concept1} and {concept2}?",
                "What factors influence {concept}?",
                "What are the implications of {concept}?"
            ],
            "comparison": [
                "Compare {concept1} and {concept2}",
                "What are the pros and cons of {concept}?",
                "How do different approaches to {concept} compare?",
                "Which is more effective: {concept1} or {concept2}?",
                "What are the similarities between {concept1} and {concept2}?"
            ],
            "explanation": [
                "Explain {concept} in simple terms",
                "How does {concept} work?",
                "What is the purpose of {concept}?",
                "Why is {concept} important?",
                "What are the components of {concept}?"
            ],
            "application": [
                "How can {concept} be applied to {domain}?",
                "What are real-world examples of {concept}?",
                "How is {concept} implemented in practice?",
                "What are the best practices for {concept}?",
                "What tools are used for {concept}?"
            ],
            "trend": [
                "What are the current trends in {concept}?",
                "How has {concept} evolved over time?",
                "What is the future of {concept}?",
                "What are emerging developments in {concept}?",
                "What challenges does {concept} face?"
            ]
        }
    
    def enhance_query(self, query: str, documents: List[Document], 
                     conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Enhance a user query with expansion, context, and suggestions."""
        # Store query in history
        self._update_query_history(query)
        
        # Analyze query intent and type
        query_analysis = self._analyze_query_intent(query)
        
        # Extract key concepts from query
        query_concepts = self._extract_query_concepts(query)
        
        # Expand query with synonyms and related terms
        expanded_query = self._expand_query_terms(query, documents)
        
        # Generate contextual suggestions
        suggestions = self._generate_intelligent_suggestions(
            query, documents, conversation_history
        )
        
        # Create follow-up questions
        follow_ups = self._generate_follow_up_questions(query, query_concepts, documents)
        
        # Analyze query complexity and provide guidance
        complexity_analysis = self._analyze_query_complexity(query)
        
        # Generate alternative phrasings
        alternatives = self._generate_query_alternatives(query, query_concepts)
        
        return {
            "original_query": query,
            "query_analysis": query_analysis,
            "extracted_concepts": query_concepts,
            "expanded_query": expanded_query,
            "intelligent_suggestions": suggestions,
            "follow_up_questions": follow_ups,
            "complexity_analysis": complexity_analysis,
            "alternative_phrasings": alternatives,
            "enhancement_timestamp": datetime.now().isoformat()
        }
    
    def _update_query_history(self, query: str):
        """Update query history for learning user patterns."""
        self.query_history.append({
            "query": query,
            "timestamp": datetime.now(),
            "concepts": self._extract_query_concepts(query)
        })
        
        # Keep only recent history (last 50 queries)
        if len(self.query_history) > 50:
            self.query_history = self.query_history[-50:]
        
        # Update user interests
        concepts = self._extract_query_concepts(query)
        for concept in concepts:
            self.user_interests[concept.lower()] += 1
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and type of the query."""
        query_lower = query.lower()
        
        # Define intent patterns
        intent_patterns = {
            "definition": [r"what is", r"define", r"meaning of", r"definition"],
            "explanation": [r"how does", r"explain", r"why", r"how to"],
            "comparison": [r"compare", r"difference", r"versus", r"vs", r"better"],
            "listing": [r"list", r"what are", r"types of", r"kinds of"],
            "analysis": [r"analyze", r"examine", r"evaluate", r"assess"],
            "application": [r"how to use", r"apply", r"implement", r"practice"],
            "opinion": [r"should", r"recommend", r"suggest", r"best"],
            "factual": [r"when", r"where", r"who", r"which", r"statistics"]
        }
        
        # Score each intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                intent_scores[intent] = score
        
        # Determine primary intent
        primary_intent = max(intent_scores.keys(), key=intent_scores.get) if intent_scores else "general"
        
        # Analyze query characteristics
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        has_question_word = any(word in query_lower for word in question_words)
        
        # Determine complexity level
        word_count = len(query.split())
        if word_count <= 3:
            complexity = "simple"
        elif word_count <= 8:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        return {
            "primary_intent": primary_intent,
            "intent_scores": intent_scores,
            "has_question_word": has_question_word,
            "is_question": query.strip().endswith('?') or has_question_word,
            "complexity": complexity,
            "word_count": word_count,
            "estimated_difficulty": self._estimate_query_difficulty(query)
        }
    
    def _estimate_query_difficulty(self, query: str) -> str:
        """Estimate the difficulty level of answering the query."""
        difficulty_indicators = {
            "basic": ["what is", "define", "list", "name"],
            "intermediate": ["how", "why", "compare", "explain"],
            "advanced": ["analyze", "evaluate", "synthesize", "critique", "implications"]
        }
        
        query_lower = query.lower()
        scores = {"basic": 0, "intermediate": 0, "advanced": 0}
        
        for level, indicators in difficulty_indicators.items():
            scores[level] = sum(1 for indicator in indicators if indicator in query_lower)
        
        max_level = max(scores.keys(), key=scores.get)
        return max_level if scores[max_level] > 0 else "intermediate"
    
    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query."""
        # Remove common question words and articles
        stop_words = {"what", "how", "why", "when", "where", "who", "which", "is", "are", 
                     "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Extract noun phrases (simple approach)
        noun_phrases = []
        for i in range(len(concepts) - 1):
            phrase = f"{concepts[i]} {concepts[i + 1]}"
            noun_phrases.append(phrase)
        
        # Combine single words and phrases
        all_concepts = concepts + noun_phrases
        
        # Return most relevant concepts (limit to avoid noise)
        return all_concepts[:10]
    
    def _expand_query_terms(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Expand query terms with synonyms and related concepts."""
        concepts = self._extract_query_concepts(query)
        
        # Build document vocabulary
        doc_text = " ".join([doc.page_content for doc in documents])
        doc_words = set(re.findall(r'\b[a-zA-Z]+\b', doc_text.lower()))
        
        # Find related terms using simple co-occurrence
        expanded_terms = {}
        
        for concept in concepts:
            related_terms = []
            
            # Find terms that frequently appear near the concept
            pattern = rf'\b\w+\s+{re.escape(concept)}\b|\b{re.escape(concept)}\s+\w+\b'
            matches = re.findall(pattern, doc_text.lower())
            
            related_words = []
            for match in matches:
                words = match.split()
                related_words.extend([w for w in words if w != concept and len(w) > 3])
            
            if related_words:
                word_freq = Counter(related_words)
                related_terms = [word for word, freq in word_freq.most_common(5) if freq > 1]
            
            if related_terms:
                expanded_terms[concept] = related_terms
        
        # Create expanded query
        expanded_parts = [query]
        for concept, related in expanded_terms.items():
            if related:
                expanded_parts.append(f"({concept} OR {' OR '.join(related[:3])})")
        
        return {
            "expanded_terms": expanded_terms,
            "expanded_query": " ".join(expanded_parts),
            "expansion_strategy": "co-occurrence_based"
        }
    
    def _generate_intelligent_suggestions(self, query: str, documents: List[Document], 
                                        conversation_history: List[Dict] = None) -> List[Dict[str, Any]]:
        """Generate intelligent query suggestions based on context."""
        suggestions = []
        
        # Extract document themes for context-aware suggestions
        doc_themes = self._extract_document_themes(documents)
        query_concepts = self._extract_query_concepts(query)
        
        # Context-based suggestions
        for theme, keywords in doc_themes.items():
            relevance_score = len(set(query_concepts) & set(keywords)) / max(len(query_concepts), 1)
            
            if relevance_score > 0.1:  # Some relevance threshold
                suggestions.append({
                    "suggestion": f"Explore {theme} in relation to your query",
                    "type": "context_expansion",
                    "relevance_score": round(relevance_score, 3),
                    "rationale": f"Your query relates to {theme} concepts in the documents"
                })
        
        # History-based suggestions
        if conversation_history:
            recent_topics = self._extract_recent_topics(conversation_history)
            for topic in recent_topics[:3]:
                if topic not in query.lower():
                    suggestions.append({
                        "suggestion": f"How does this relate to {topic} from earlier?",
                        "type": "conversation_continuity",
                        "relevance_score": 0.8,
                        "rationale": "Building on previous conversation topics"
                    })
        
        # User interest-based suggestions
        top_interests = sorted(self.user_interests.items(), key=lambda x: x[1], reverse=True)[:5]
        for interest, count in top_interests:
            if interest not in query.lower() and count > 2:
                suggestions.append({
                    "suggestion": f"Connect this to {interest} (frequent interest)",
                    "type": "personal_interest",
                    "relevance_score": min(1.0, count / 10),
                    "rationale": f"You've shown interest in {interest} in past queries"
                })
        
        # Document-specific suggestions
        doc_specific = self._generate_document_specific_suggestions(query, documents)
        suggestions.extend(doc_specific)
        
        # Sort by relevance and limit
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        return suggestions[:8]
    
    def _extract_document_themes(self, documents: List[Document]) -> Dict[str, List[str]]:
        """Extract main themes from documents."""
        if not documents:
            return {}
        
        # Combine all document text
        all_text = " ".join([doc.page_content for doc in documents])
        
        try:
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([all_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = scores.argsort()[-20:][::-1]
            top_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            # Group terms into themes (simple clustering by co-occurrence)
            themes = {"main_concepts": top_terms[:10]}
            
            # Add some basic theme detection
            if any(term in all_text.lower() for term in ['research', 'study', 'analysis']):
                themes["research"] = [term for term in top_terms if term in all_text.lower()]
            
            if any(term in all_text.lower() for term in ['technology', 'system', 'algorithm']):
                themes["technology"] = [term for term in top_terms if term in all_text.lower()]
            
            return themes
            
        except Exception:
            # Fallback: simple word frequency
            words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
            word_freq = Counter(w for w in words if len(w) > 4)
            return {"main_concepts": [word for word, freq in word_freq.most_common(10)]}
    
    def _extract_recent_topics(self, conversation_history: List[Dict]) -> List[str]:
        """Extract recent topics from conversation history."""
        topics = []
        
        for exchange in conversation_history[-5:]:  # Last 5 exchanges
            question = exchange.get('question', '')
            concepts = self._extract_query_concepts(question)
            topics.extend(concepts)
        
        # Return most frequent recent topics
        topic_freq = Counter(topics)
        return [topic for topic, freq in topic_freq.most_common(5)]
    
    def _generate_document_specific_suggestions(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        """Generate suggestions specific to document content."""
        suggestions = []
        
        # Find documents most relevant to query
        try:
            texts = [doc.page_content for doc in documents]
            vectorizer = TfidfVectorizer(stop_words='english')
            doc_vectors = vectorizer.fit_transform(texts)
            query_vector = vectorizer.transform([query])
            
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            top_doc_indices = similarities.argsort()[-3:][::-1]  # Top 3 most relevant docs
            
            for idx in top_doc_indices:
                if similarities[idx] > 0.1:  # Some similarity threshold
                    doc = documents[idx]
                    filename = doc.metadata.get('filename', f'Document_{idx}')
                    
                    suggestions.append({
                        "suggestion": f"Explore more details from {filename}",
                        "type": "document_deep_dive",
                        "relevance_score": round(float(similarities[idx]), 3),
                        "rationale": f"This document appears highly relevant to your query"
                    })
                    
        except Exception:
            pass  # Fallback gracefully
        
        return suggestions
    
    def _generate_follow_up_questions(self, query: str, concepts: List[str], 
                                    documents: List[Document]) -> List[Dict[str, Any]]:
        """Generate intelligent follow-up questions."""
        follow_ups = []
        
        # Analyze query intent for appropriate follow-ups
        query_analysis = self._analyze_query_intent(query)
        primary_intent = query_analysis['primary_intent']
        
        # Get document themes for context
        doc_themes = self._extract_document_themes(documents)
        main_concepts = doc_themes.get('main_concepts', concepts)
        
        # Generate follow-ups based on intent
        if primary_intent == 'definition':
            follow_ups.extend([
                {"question": f"How is {concepts[0]} implemented in practice?", "type": "application"},
                {"question": f"What are the benefits of {concepts[0]}?", "type": "analysis"},
                {"question": f"What challenges are associated with {concepts[0]}?", "type": "analysis"}
            ])
        
        elif primary_intent == 'comparison':
            follow_ups.extend([
                {"question": f"What are the use cases for each approach?", "type": "application"},
                {"question": f"Which factors should guide the choice?", "type": "decision"},
                {"question": f"Are there hybrid approaches combining both?", "type": "exploration"}
            ])
        
        elif primary_intent == 'explanation':
            follow_ups.extend([
                {"question": f"What are real-world examples?", "type": "examples"},
                {"question": f"What are the limitations?", "type": "analysis"},
                {"question": f"How has this evolved over time?", "type": "historical"}
            ])
        
        # Add concept-specific follow-ups
        for concept in concepts[:2]:  # Limit to avoid too many suggestions
            if concept in " ".join(main_concepts):
                follow_ups.append({
                    "question": f"What are the latest developments in {concept}?",
                    "type": "trending"
                })
                follow_ups.append({
                    "question": f"How does {concept} relate to other concepts in the documents?",
                    "type": "relationship"
                })
        
        # Add document-specific follow-ups
        if len(documents) > 1:
            follow_ups.append({
                "question": "Are there any contradictions between the documents on this topic?",
                "type": "contradiction_check"
            })
            follow_ups.append({
                "question": "Which document provides the most comprehensive coverage?",
                "type": "comparison"
            })
        
        # Limit and randomize
        import random
        random.shuffle(follow_ups)
        return follow_ups[:6]
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and provide guidance."""
        words = query.split()
        
        # Complexity indicators
        complex_words = [w for w in words if len(w) > 8]
        technical_terms = [w for w in words if any(char.isupper() for char in w[1:])]  # CamelCase or acronyms
        question_depth = len([w for w in words if w.lower() in ['why', 'how', 'analyze', 'evaluate']])
        
        # Calculate complexity score
        complexity_score = (
            len(words) * 0.1 +
            len(complex_words) * 0.3 +
            len(technical_terms) * 0.2 +
            question_depth * 0.4
        )
        
        # Classify complexity
        if complexity_score < 2:
            complexity_level = "simple"
            guidance = "Your query is straightforward - expect direct answers"
        elif complexity_score < 5:
            complexity_level = "moderate"
            guidance = "This is a moderately complex query - may require multiple perspectives"
        else:
            complexity_level = "complex"
            guidance = "This is a complex query - consider breaking it into smaller questions"
        
        return {
            "complexity_score": round(complexity_score, 2),
            "complexity_level": complexity_level,
            "guidance": guidance,
            "suggestions": self._get_complexity_suggestions(complexity_level, query)
        }
    
    def _get_complexity_suggestions(self, complexity_level: str, query: str) -> List[str]:
        """Get suggestions based on query complexity."""
        if complexity_level == "simple":
            return [
                "Consider asking follow-up questions for deeper understanding",
                "You might want to explore related concepts"
            ]
        elif complexity_level == "moderate":
            return [
                "Good balance of specificity and scope",
                "Consider specifying context if needed"
            ]
        else:  # complex
            return [
                "Consider breaking this into smaller, focused questions",
                "Try asking about specific aspects separately",
                "Start with fundamental concepts before advanced analysis"
            ]
    
    def _generate_query_alternatives(self, query: str, concepts: List[str]) -> List[Dict[str, str]]:
        """Generate alternative ways to phrase the query."""
        alternatives = []
        
        # Simple rephrasing patterns
        if query.startswith("What is"):
            alternatives.append({
                "alternative": query.replace("What is", "Can you explain"),
                "style": "conversational"
            })
            alternatives.append({
                "alternative": query.replace("What is", "Define"),
                "style": "direct"
            })
        
        if "how" in query.lower():
            alternatives.append({
                "alternative": query.replace("How", "In what way"),
                "style": "formal"
            })
        
        # Add concept-based alternatives
        if concepts:
            main_concept = concepts[0]
            alternatives.append({
                "alternative": f"Tell me about {main_concept}",
                "style": "simple"
            })
            alternatives.append({
                "alternative": f"Provide an overview of {main_concept}",
                "style": "academic"
            })
        
        # Template-based alternatives
        query_lower = query.lower()
        if "compare" in query_lower:
            alternatives.append({
                "alternative": query.replace("compare", "what are the differences between"),
                "style": "explicit"
            })
        
        return alternatives[:4]  # Limit alternatives
    
    def get_trending_queries(self, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Get trending queries from recent history."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_queries = [q for q in self.query_history if q['timestamp'] > cutoff_time]
        
        if not recent_queries:
            return []
        
        # Count concept frequency
        concept_freq = Counter()
        for query_data in recent_queries:
            for concept in query_data['concepts']:
                concept_freq[concept.lower()] += 1
        
        trending = []
        for concept, freq in concept_freq.most_common(10):
            if freq > 1:  # Must appear more than once
                trending.append({
                    "concept": concept,
                    "frequency": freq,
                    "trend_score": freq / len(recent_queries),
                    "sample_query": next(q['query'] for q in recent_queries 
                                       if concept in [c.lower() for c in q['concepts']])
                })
        
        return trending
    
    def suggest_exploration_paths(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Suggest exploration paths through the document corpus."""
        themes = self._extract_document_themes(documents)
        paths = []
        
        # Create learning paths based on document themes
        for theme, keywords in themes.items():
            if len(keywords) >= 3:
                path = {
                    "theme": theme,
                    "suggested_sequence": [
                        f"Start with: What is {keywords[0]}?",
                        f"Then explore: How does {keywords[0]} relate to {keywords[1]}?",
                        f"Deep dive: What are advanced applications of {keywords[0]}?",
                        f"Synthesis: How do {keywords[0]} and {keywords[1]} work together?"
                    ],
                    "estimated_time": "15-20 minutes",
                    "difficulty": "progressive"
                }
                paths.append(path)
        
        return paths[:3]  # Top 3 paths