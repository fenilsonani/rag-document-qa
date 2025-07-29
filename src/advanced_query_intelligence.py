"""
Advanced Query Intelligence System - Pro-Level Enhancement
Implements sophisticated intent classification, query rewriting, and adaptive reasoning.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Some advanced query intelligence features will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Semantic query analysis will be limited.")

from langchain.schema import Document
from .config import Config


@dataclass
class QueryIntent:
    """Represents a classified query intent."""
    primary_intent: str
    confidence: float
    sub_intents: Dict[str, float]
    reasoning_type: str
    complexity_level: str
    domain_context: Optional[str] = None
    temporal_aspect: Optional[str] = None
    
    
@dataclass
class QueryRewrite:
    """Represents a rewritten query with metadata."""
    original_query: str
    rewritten_query: str
    rewrite_type: str
    confidence: float
    reasoning: str
    improvement_factors: List[str]


class IntentClassifier:
    """Advanced intent classification using multiple approaches."""
    
    def __init__(self):
        self.config = Config()
        
        # Define comprehensive intent taxonomy
        self.intent_taxonomy = {
            # Primary intents
            "factual_retrieval": {
                "patterns": [r"what is", r"define", r"definition of", r"who is", r"when did", r"where is"],
                "keywords": ["definition", "meaning", "who", "what", "when", "where"],
                "description": "Seeking factual information or definitions"
            },
            "procedural": {
                "patterns": [r"how to", r"how do", r"steps to", r"process of", r"method to"],
                "keywords": ["how", "steps", "process", "method", "procedure", "tutorial"],
                "description": "Asking for instructions or procedures"
            },
            "analytical": {
                "patterns": [r"analyze", r"examine", r"evaluate", r"assess", r"investigate"],
                "keywords": ["analyze", "examine", "evaluate", "assess", "why", "causes", "reasons"],
                "description": "Requesting analysis or evaluation"
            },
            "comparative": {
                "patterns": [r"compare", r"contrast", r"difference", r"versus", r"vs", r"better than"],
                "keywords": ["compare", "contrast", "difference", "versus", "vs", "better", "worse"],
                "description": "Comparing multiple concepts or options"
            },
            "causal": {
                "patterns": [r"why does", r"what causes", r"reason for", r"because of", r"due to"],
                "keywords": ["why", "causes", "reason", "because", "due", "leads", "results"],
                "description": "Understanding cause-effect relationships"
            },
            "predictive": {
                "patterns": [r"what will", r"predict", r"forecast", r"future", r"trend"],
                "keywords": ["will", "predict", "future", "trend", "forecast", "expect", "likely"],
                "description": "Making predictions or understanding trends"
            },
            "creative": {
                "patterns": [r"generate", r"create", r"brainstorm", r"ideas for", r"suggest"],
                "keywords": ["generate", "create", "brainstorm", "ideas", "suggest", "propose"],
                "description": "Generating new ideas or creative solutions"
            },
            "troubleshooting": {
                "patterns": [r"problem with", r"error", r"fix", r"solve", r"debug", r"troubleshoot"],
                "keywords": ["problem", "error", "fix", "solve", "debug", "issue", "bug"],
                "description": "Solving problems or debugging issues"
            },
            "verification": {
                "patterns": [r"is it true", r"verify", r"confirm", r"validate", r"check"],
                "keywords": ["true", "verify", "confirm", "validate", "check", "correct"],
                "description": "Verifying or confirming information"
            },
            "optimization": {
                "patterns": [r"optimize", r"improve", r"enhance", r"best way", r"most efficient"],
                "keywords": ["optimize", "improve", "enhance", "best", "efficient", "performance"],
                "description": "Seeking improvements or optimizations"
            }
        }
        
        # Reasoning type classification
        self.reasoning_types = {
            "deductive": ["therefore", "thus", "consequently", "given that", "since"],
            "inductive": ["pattern", "trend", "generally", "typically", "usually"],
            "abductive": ["best explanation", "most likely", "probably", "hypothesis"],
            "analogical": ["similar to", "like", "compared to", "analogous"],
            "temporal": ["before", "after", "during", "timeline", "sequence"],
            "spatial": ["where", "location", "position", "geography", "distance"]
        }
        
        # Initialize ML components if available
        self.ml_classifier = None
        self.sentence_model = None
        self._initialize_ml_components()
        
        # Training data for intent classification (if ML is available)
        self.training_examples = self._load_training_data()
        
    def _initialize_ml_components(self):
        """Initialize machine learning components."""
        if SKLEARN_AVAILABLE:
            try:
                # Simple pipeline for intent classification
                self.ml_classifier = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
                    ('classifier', MultinomialNB(alpha=0.1))
                ])
                logging.info("ML classifier initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize ML classifier: {e}")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info("Sentence transformer initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize sentence transformer: {e}")
    
    def _load_training_data(self) -> List[Tuple[str, str]]:
        """Load training data for intent classification."""
        # This would typically be loaded from a file or database
        # For now, we'll use a small hardcoded dataset
        return [
            ("What is machine learning?", "factual_retrieval"),
            ("How do I implement a neural network?", "procedural"),
            ("Compare deep learning and traditional ML", "comparative"),
            ("Why does overfitting occur?", "causal"),
            ("Analyze the performance of this model", "analytical"),
            ("What will be the future of AI?", "predictive"),
            ("Generate ideas for improving accuracy", "creative"),
            ("Fix the error in my training loop", "troubleshooting"),
            ("Is this result statistically significant?", "verification"),
            ("Optimize my model for faster inference", "optimization")
        ]
    
    def classify_intent(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryIntent:
        """Classify the intent of a query using multiple approaches."""
        query_lower = query.lower().strip()
        
        # Rule-based classification
        rule_based_results = self._rule_based_classification(query_lower)
        
        # ML-based classification (if available)
        ml_results = self._ml_based_classification(query) if self.ml_classifier else {}
        
        # Semantic classification (if available)
        semantic_results = self._semantic_classification(query) if self.sentence_model else {}
        
        # Combine results with weighted voting
        combined_scores = self._combine_classification_results(
            rule_based_results, ml_results, semantic_results
        )
        
        # Determine primary intent
        if combined_scores:
            primary_intent = max(combined_scores.keys(), key=combined_scores.get)
            confidence = combined_scores[primary_intent]
        else:
            primary_intent = "general"
            confidence = 0.5
        
        # Classify reasoning type
        reasoning_type = self._classify_reasoning_type(query_lower)
        
        # Determine complexity level
        complexity_level = self._determine_complexity(query)
        
        # Extract domain context
        domain_context = self._extract_domain_context(query, context)
        
        # Detect temporal aspects
        temporal_aspect = self._detect_temporal_aspect(query_lower)
        
        return QueryIntent(
            primary_intent=primary_intent,
            confidence=confidence,
            sub_intents=combined_scores,
            reasoning_type=reasoning_type,
            complexity_level=complexity_level,
            domain_context=domain_context,
            temporal_aspect=temporal_aspect
        )
    
    def _rule_based_classification(self, query: str) -> Dict[str, float]:
        """Rule-based intent classification."""
        scores = defaultdict(float)
        
        for intent, config in self.intent_taxonomy.items():
            # Pattern matching
            pattern_score = 0
            for pattern in config["patterns"]:
                if re.search(pattern, query):
                    pattern_score += 1
            
            # Keyword matching
            keyword_score = 0
            for keyword in config["keywords"]:
                if keyword in query:
                    keyword_score += 1
            
            # Combined score with normalization
            total_score = (pattern_score * 2 + keyword_score) / (len(config["patterns"]) + len(config["keywords"]))
            scores[intent] = total_score
        
        return dict(scores)
    
    def _ml_based_classification(self, query: str) -> Dict[str, float]:
        """ML-based intent classification."""
        if not self.ml_classifier:
            return {}
        
        try:
            # Train classifier if not already done
            if not hasattr(self.ml_classifier, 'classes_'):
                queries, labels = zip(*self.training_examples)
                self.ml_classifier.fit(queries, labels)
            
            # Predict probabilities
            probabilities = self.ml_classifier.predict_proba([query])[0]
            classes = self.ml_classifier.classes_
            
            return {classes[i]: prob for i, prob in enumerate(probabilities)}
            
        except Exception as e:
            logging.warning(f"ML classification failed: {e}")
            return {}
    
    def _semantic_classification(self, query: str) -> Dict[str, float]:
        """Semantic intent classification using sentence embeddings."""
        if not self.sentence_model:
            return {}
        
        try:
            # Create prototype queries for each intent
            intent_prototypes = {
                "factual_retrieval": "What is the definition of this concept?",
                "procedural": "How do I do this step by step?",
                "analytical": "Analyze and evaluate this topic in detail.",
                "comparative": "Compare these different options and approaches.",
                "causal": "Why does this happen and what causes it?",
                "predictive": "What will happen in the future with this?",
                "creative": "Generate new ideas and creative solutions.",
                "troubleshooting": "Fix this problem and solve the issue.",
                "verification": "Is this information true and accurate?",
                "optimization": "How can this be improved and optimized?"
            }
            
            # Compute embeddings
            query_embedding = self.sentence_model.encode([query])
            prototype_embeddings = self.sentence_model.encode(list(intent_prototypes.values()))
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, prototype_embeddings)[0]
            
            # Convert to scores
            intent_names = list(intent_prototypes.keys())
            scores = {intent_names[i]: float(sim) for i, sim in enumerate(similarities)}
            
            return scores
            
        except Exception as e:
            logging.warning(f"Semantic classification failed: {e}")
            return {}
    
    def _combine_classification_results(self, rule_based: Dict[str, float], 
                                      ml_based: Dict[str, float], 
                                      semantic: Dict[str, float]) -> Dict[str, float]:
        """Combine classification results with weighted voting."""
        # Weights for different approaches
        weights = {
            'rule_based': 0.4,
            'ml_based': 0.3,
            'semantic': 0.3
        }
        
        all_intents = set(rule_based.keys()) | set(ml_based.keys()) | set(semantic.keys())
        combined_scores = {}
        
        for intent in all_intents:
            score = 0.0
            
            # Rule-based contribution
            if intent in rule_based:
                score += rule_based[intent] * weights['rule_based']
            
            # ML-based contribution
            if intent in ml_based:
                score += ml_based[intent] * weights['ml_based']
            
            # Semantic contribution
            if intent in semantic:
                score += semantic[intent] * weights['semantic']
            
            combined_scores[intent] = score
        
        # Normalize scores
        if combined_scores:
            max_score = max(combined_scores.values())
            if max_score > 0:
                combined_scores = {k: v / max_score for k, v in combined_scores.items()}
        
        return combined_scores
    
    def _classify_reasoning_type(self, query: str) -> str:
        """Classify the type of reasoning required."""
        for reasoning_type, indicators in self.reasoning_types.items():
            if any(indicator in query for indicator in indicators):
                return reasoning_type
        
        # Default reasoning type based on query structure
        if any(word in query for word in ["why", "because", "cause"]):
            return "causal"
        elif any(word in query for word in ["how", "what", "where"]):
            return "deductive"
        else:
            return "general"
    
    def _determine_complexity(self, query: str) -> str:
        """Determine the complexity level of the query."""
        # Complexity indicators
        words = query.split()
        word_count = len(words)
        
        # Count complex indicators
        complex_words = [w for w in words if len(w) > 8]
        technical_terms = [w for w in words if any(char.isupper() for char in w[1:])]
        question_depth_words = [w for w in words if w.lower() in 
                               ['analyze', 'evaluate', 'synthesize', 'compare', 'contrast']]
        
        # Calculate complexity score
        complexity_score = (
            min(word_count / 10, 1.0) * 0.3 +
            min(len(complex_words) / 3, 1.0) * 0.3 +
            min(len(technical_terms) / 2, 1.0) * 0.2 +
            min(len(question_depth_words) / 2, 1.0) * 0.2
        )
        
        if complexity_score < 0.3:
            return "simple"
        elif complexity_score < 0.7:
            return "moderate"
        else:
            return "complex"
    
    def _extract_domain_context(self, query: str, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract domain context from query and additional context."""
        # Domain keywords
        domains = {
            "technology": ["software", "algorithm", "code", "programming", "system", "technical"],
            "science": ["research", "study", "experiment", "hypothesis", "theory", "scientific"],
            "business": ["market", "strategy", "profit", "revenue", "customer", "business"],
            "healthcare": ["medical", "health", "patient", "treatment", "diagnosis", "clinical"],
            "education": ["learning", "teaching", "student", "curriculum", "academic", "educational"],
            "finance": ["money", "investment", "financial", "cost", "budget", "economic"]
        }
        
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Also check context if provided
        if context and 'document_domains' in context:
            for domain in context['document_domains']:
                domain_scores[domain] = domain_scores.get(domain, 0) + 1
        
        return max(domain_scores.keys(), key=domain_scores.get) if domain_scores else None
    
    def _detect_temporal_aspect(self, query: str) -> Optional[str]:
        """Detect temporal aspects in the query."""
        temporal_indicators = {
            "past": ["was", "were", "previously", "before", "historical", "past"],
            "present": ["is", "are", "currently", "now", "present", "today"],
            "future": ["will", "future", "predict", "forecast", "upcoming", "next"]
        }
        
        for temporal_type, indicators in temporal_indicators.items():
            if any(indicator in query for indicator in indicators):
                return temporal_type
        
        return None


class QueryRewriter:
    """Advanced query rewriting system for optimization and clarity."""
    
    def __init__(self):
        self.config = Config()
        
        # Rewriting strategies
        self.rewrite_strategies = {
            "clarification": self._clarify_ambiguous_terms,
            "expansion": self._expand_with_context,
            "simplification": self._simplify_complex_query,
            "specificity": self._add_specificity,
            "semantic_enhancement": self._enhance_semantics,
            "structural_improvement": self._improve_structure
        }
        
        # Ambiguous terms that often need clarification
        self.ambiguous_terms = {
            "it": ["the system", "the method", "the concept", "the approach"],
            "this": ["this concept", "this method", "this approach", "this system"],
            "that": ["that concept", "that method", "that approach", "that system"],
            "they": ["these systems", "these methods", "these concepts", "these approaches"]
        }
        
        # Common query patterns that can be improved
        self.improvement_patterns = [
            (r"how to (.+)", r"What are the steps to \1?"),
            (r"what is (.+)", r"Can you explain \1 and its significance?"),
            (r"why (.+)", r"What are the reasons and underlying causes for \1?"),
            (r"compare (.+) and (.+)", r"What are the key differences and similarities between \1 and \2?")
        ]
    
    def rewrite_query(self, query: str, intent: QueryIntent, 
                     context: Optional[Dict[str, Any]] = None) -> List[QueryRewrite]:
        """Generate multiple rewritten versions of the query."""
        rewrites = []
        
        # Apply different rewriting strategies
        for strategy_name, strategy_func in self.rewrite_strategies.items():
            try:
                rewritten = strategy_func(query, intent, context)
                if rewritten and rewritten != query:
                    confidence = self._calculate_rewrite_confidence(query, rewritten, intent)
                    improvement_factors = self._identify_improvements(query, rewritten)
                    
                    rewrite = QueryRewrite(
                        original_query=query,
                        rewritten_query=rewritten,
                        rewrite_type=strategy_name,
                        confidence=confidence,
                        reasoning=self._generate_rewrite_reasoning(strategy_name, query, rewritten),
                        improvement_factors=improvement_factors
                    )
                    rewrites.append(rewrite)
                    
            except Exception as e:
                logging.warning(f"Rewrite strategy {strategy_name} failed: {e}")
                continue
        
        # Sort by confidence and return top rewrites
        rewrites.sort(key=lambda x: x.confidence, reverse=True)
        return rewrites[:3]  # Return top 3 rewrites
    
    def _clarify_ambiguous_terms(self, query: str, intent: QueryIntent, 
                                context: Optional[Dict[str, Any]]) -> str:
        """Clarify ambiguous terms in the query."""
        clarified_query = query
        
        # Replace ambiguous pronouns
        for ambiguous, replacements in self.ambiguous_terms.items():
            if ambiguous in query.lower():
                # Choose replacement based on context or default to first option
                replacement = replacements[0]
                if context and 'main_concept' in context:
                    replacement = f"the {context['main_concept']}"
                
                clarified_query = re.sub(rf'\b{ambiguous}\b', replacement, clarified_query, flags=re.IGNORECASE)
        
        return clarified_query
    
    def _expand_with_context(self, query: str, intent: QueryIntent, 
                           context: Optional[Dict[str, Any]]) -> str:
        """Expand query with relevant context."""
        if not context or 'document_themes' not in context:
            return query
        
        themes = context['document_themes']
        main_theme = max(themes.keys(), key=lambda k: len(themes[k])) if themes else None
        
        if main_theme and main_theme.lower() not in query.lower():
            if intent.primary_intent == "factual_retrieval":
                return f"{query} in the context of {main_theme}"
            elif intent.primary_intent == "analytical":
                return f"Analyze {query} specifically in relation to {main_theme}"
            else:
                return f"{query} (focusing on {main_theme} aspects)"
        
        return query
    
    def _simplify_complex_query(self, query: str, intent: QueryIntent, 
                               context: Optional[Dict[str, Any]]) -> str:
        """Simplify overly complex queries."""
        if intent.complexity_level != "complex":
            return query
        
        # Break down complex queries into main components
        words = query.split()
        if len(words) > 15:
            # Find the main verb or question word
            question_words = ["what", "how", "why", "when", "where", "which"]
            main_parts = []
            
            for word in words:
                if word.lower() in question_words or word.endswith('?'):
                    # Take this word and the next few words as the core question
                    word_index = words.index(word)
                    main_parts = words[word_index:word_index + 8]
                    break
            
            if main_parts:
                return " ".join(main_parts)
        
        return query
    
    def _add_specificity(self, query: str, intent: QueryIntent, 
                        context: Optional[Dict[str, Any]]) -> str:
        """Add specificity to vague queries."""
        specificity_additions = {
            "factual_retrieval": ["specifically", "in detail"],
            "analytical": ["comprehensively", "from multiple perspectives"],
            "comparative": ["systematically", "across key dimensions"],
            "procedural": ["step-by-step", "with practical examples"]
        }
        
        additions = specificity_additions.get(intent.primary_intent, [])
        if additions and not any(add in query.lower() for add in additions):
            return f"{query} {additions[0]}"
        
        return query
    
    def _enhance_semantics(self, query: str, intent: QueryIntent, 
                          context: Optional[Dict[str, Any]]) -> str:
        """Enhance semantic clarity of the query."""
        # Apply pattern-based improvements
        enhanced_query = query
        
        for pattern, replacement in self.improvement_patterns:
            enhanced_query = re.sub(pattern, replacement, enhanced_query, flags=re.IGNORECASE)
        
        # Add semantic markers based on intent
        if intent.primary_intent == "causal" and "because" not in enhanced_query.lower():
            enhanced_query = f"What are the underlying reasons why {enhanced_query.lower()}"
            
        elif intent.primary_intent == "predictive" and "will" not in enhanced_query.lower():
            enhanced_query = f"What are the likely future developments regarding {enhanced_query.lower()}"
        
        return enhanced_query
    
    def _improve_structure(self, query: str, intent: QueryIntent, 
                          context: Optional[Dict[str, Any]]) -> str:
        """Improve the structural clarity of the query."""
        # Ensure proper question structure
        if not query.strip().endswith('?') and intent.primary_intent != "creative":
            # Add question mark if it's clearly a question
            question_indicators = ["what", "how", "why", "when", "where", "which", "who"]
            if any(query.lower().startswith(word) for word in question_indicators):
                return f"{query}?"
        
        # Improve sentence structure for complex queries
        if intent.complexity_level == "complex":
            # Add connecting words for better flow
            if " and " in query and "as well as" not in query:
                query = query.replace(" and ", " as well as ", 1)
        
        return query
    
    def _calculate_rewrite_confidence(self, original: str, rewritten: str, intent: QueryIntent) -> float:
        """Calculate confidence in the rewrite quality."""
        # Base confidence on various factors
        confidence_factors = []
        
        # Length improvement (moderate increase is good)
        len_ratio = len(rewritten) / len(original)
        if 1.1 <= len_ratio <= 2.0:
            confidence_factors.append(0.8)
        elif len_ratio > 2.0:
            confidence_factors.append(0.5)  # Too much expansion
        else:
            confidence_factors.append(0.6)
        
        # Structural improvements
        if rewritten.endswith('?') and not original.endswith('?'):
            confidence_factors.append(0.9)
        
        # Specificity improvements
        specific_words = ["specifically", "detailed", "comprehensive", "systematic"]
        if any(word in rewritten.lower() for word in specific_words):
            confidence_factors.append(0.8)
        
        # Avoid over-complication
        if intent.complexity_level == "simple" and len(rewritten.split()) > len(original.split()) * 1.5:
            confidence_factors.append(0.4)
        
        return np.mean(confidence_factors) if confidence_factors else 0.7
    
    def _identify_improvements(self, original: str, rewritten: str) -> List[str]:
        """Identify specific improvements made in the rewrite."""
        improvements = []
        
        # Check for structural improvements
        if rewritten.endswith('?') and not original.endswith('?'):
            improvements.append("Added proper question structure")
        
        # Check for clarity improvements
        ambiguous_terms = ["it", "this", "that", "they"]
        if any(term in original.lower() for term in ambiguous_terms) and \
           not any(term in rewritten.lower() for term in ambiguous_terms):
            improvements.append("Clarified ambiguous references")
        
        # Check for specificity improvements
        if len(rewritten) > len(original) * 1.1:
            improvements.append("Added specificity and detail")
        
        # Check for context improvements
        context_words = ["context", "specifically", "regarding", "in relation to"]
        if any(word in rewritten.lower() for word in context_words):
            improvements.append("Added contextual information")
        
        return improvements
    
    def _generate_rewrite_reasoning(self, strategy: str, original: str, rewritten: str) -> str:
        """Generate reasoning for why the rewrite was made."""
        reasoning_templates = {
            "clarification": f"Clarified ambiguous terms to make the query more specific and actionable.",
            "expansion": f"Added contextual information to help retrieve more relevant results.",
            "simplification": f"Simplified complex phrasing to improve comprehension and processing.",
            "specificity": f"Enhanced specificity to target more precise information.",
            "semantic_enhancement": f"Improved semantic structure for better understanding.",
            "structural_improvement": f"Enhanced query structure for optimal processing."
        }
        
        return reasoning_templates.get(strategy, f"Applied {strategy} strategy to improve query quality.")


class AdvancedQueryIntelligence:
    """Main class combining intent classification and query rewriting."""
    
    def __init__(self):
        self.config = Config()
        self.intent_classifier = IntentClassifier()
        self.query_rewriter = QueryRewriter()
        
        # Query history and learning
        self.query_history = []
        self.performance_metrics = defaultdict(list)
        
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query with complete intelligence pipeline."""
        start_time = datetime.now()
        
        # Classify intent
        intent = self.intent_classifier.classify_intent(query, context)
        
        # Generate query rewrites
        rewrites = self.query_rewriter.rewrite_query(query, intent, context)
        
        # Select best rewrite
        best_rewrite = rewrites[0] if rewrites else None
        
        # Store in history
        self._update_history(query, intent, rewrites)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "original_query": query,
            "intent_analysis": asdict(intent),
            "query_rewrites": [asdict(rewrite) for rewrite in rewrites],
            "recommended_query": best_rewrite.rewritten_query if best_rewrite else query,
            "processing_time": processing_time,
            "enhancement_strategies": self._suggest_enhancement_strategies(intent),
            "confidence_score": intent.confidence,
            "improvement_potential": self._calculate_improvement_potential(query, intent, rewrites)
        }
    
    def _update_history(self, query: str, intent: QueryIntent, rewrites: List[QueryRewrite]):
        """Update query processing history."""
        self.query_history.append({
            "query": query,
            "intent": intent,
            "rewrites": rewrites,
            "timestamp": datetime.now()
        })
        
        # Keep only recent history
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
    
    def _suggest_enhancement_strategies(self, intent: QueryIntent) -> List[Dict[str, str]]:
        """Suggest strategies for query enhancement based on intent."""
        strategies = []
        
        if intent.confidence < 0.7:
            strategies.append({
                "strategy": "clarification",
                "description": "Consider rephrasing the query for clearer intent",
                "example": "Instead of 'How to do this?', try 'What are the steps to implement X?'"
            })
        
        if intent.complexity_level == "complex":
            strategies.append({
                "strategy": "decomposition",
                "description": "Break down complex queries into simpler parts",
                "example": "Split multi-part questions into separate, focused queries"
            })
        
        if intent.primary_intent == "comparative" and "versus" not in intent:
            strategies.append({
                "strategy": "explicit_comparison",
                "description": "Make comparison criteria explicit",
                "example": "Specify what aspects you want to compare (performance, cost, etc.)"
            })
        
        return strategies
    
    def _calculate_improvement_potential(self, query: str, intent: QueryIntent, 
                                       rewrites: List[QueryRewrite]) -> Dict[str, float]:
        """Calculate potential for query improvement."""
        potential_scores = {}
        
        # Intent clarity potential
        potential_scores["intent_clarity"] = max(0, 1.0 - intent.confidence)
        
        # Structure improvement potential
        if not query.strip().endswith('?') and intent.primary_intent in ["factual_retrieval", "procedural"]:
            potential_scores["structure"] = 0.8
        else:
            potential_scores["structure"] = 0.2
        
        # Specificity potential
        if intent.complexity_level == "simple" and len(query.split()) < 5:
            potential_scores["specificity"] = 0.7
        else:
            potential_scores["specificity"] = 0.3
        
        # Context enhancement potential
        ambiguous_words = ["it", "this", "that", "they"]
        if any(word in query.lower() for word in ambiguous_words):
            potential_scores["context"] = 0.9
        else:
            potential_scores["context"] = 0.1
        
        return potential_scores
    
    def get_query_analytics(self) -> Dict[str, Any]:
        """Get analytics about query processing patterns."""
        if not self.query_history:
            return {"message": "No query history available"}
        
        # Intent distribution
        intent_counts = Counter(entry["intent"].primary_intent for entry in self.query_history)
        
        # Complexity distribution
        complexity_counts = Counter(entry["intent"].complexity_level for entry in self.query_history)
        
        # Average confidence
        avg_confidence = np.mean([entry["intent"].confidence for entry in self.query_history])
        
        # Rewrite success rate
        successful_rewrites = sum(1 for entry in self.query_history if entry["rewrites"])
        rewrite_success_rate = successful_rewrites / len(self.query_history)
        
        return {
            "total_queries_processed": len(self.query_history),
            "intent_distribution": dict(intent_counts),
            "complexity_distribution": dict(complexity_counts),
            "average_confidence": round(avg_confidence, 3),
            "rewrite_success_rate": round(rewrite_success_rate, 3),
            "most_common_intent": intent_counts.most_common(1)[0][0] if intent_counts else None,
            "recent_trends": self._analyze_recent_trends()
        }
    
    def _analyze_recent_trends(self) -> Dict[str, Any]:
        """Analyze recent trends in query patterns."""
        if len(self.query_history) < 10:
            return {"message": "Not enough data for trend analysis"}
        
        recent_queries = self.query_history[-10:]
        older_queries = self.query_history[-20:-10] if len(self.query_history) >= 20 else []
        
        if not older_queries:
            return {"message": "Not enough historical data"}
        
        # Compare intent distributions
        recent_intents = Counter(entry["intent"].primary_intent for entry in recent_queries)
        older_intents = Counter(entry["intent"].primary_intent for entry in older_queries)
        
        # Calculate trend changes
        trends = {}
        for intent in set(recent_intents.keys()) | set(older_intents.keys()):
            recent_pct = recent_intents[intent] / len(recent_queries)
            older_pct = older_intents[intent] / len(older_queries)
            change = recent_pct - older_pct
            
            if abs(change) > 0.1:  # Significant change threshold
                trends[intent] = {
                    "change": round(change, 3),
                    "direction": "increasing" if change > 0 else "decreasing"
                }
        
        return {
            "significant_changes": trends,
            "recent_complexity_trend": self._calculate_complexity_trend(recent_queries, older_queries)
        }
    
    def _calculate_complexity_trend(self, recent: List[Dict], older: List[Dict]) -> str:
        """Calculate trend in query complexity."""
        complexity_scores = {"simple": 1, "moderate": 2, "complex": 3}
        
        recent_avg = np.mean([complexity_scores[entry["intent"].complexity_level] for entry in recent])
        older_avg = np.mean([complexity_scores[entry["intent"].complexity_level] for entry in older])
        
        diff = recent_avg - older_avg
        if diff > 0.2:
            return "increasing_complexity"
        elif diff < -0.2:
            return "decreasing_complexity"
        else:
            return "stable"