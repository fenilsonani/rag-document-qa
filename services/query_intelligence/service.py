"""
Query Intelligence Service - Enterprise RAG Platform
Advanced query understanding, routing, and orchestration service.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from uuid import uuid4

import numpy as np
from fastapi import HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..base.service_base import BaseService, ServiceStatus, ServiceRequest, ServiceResponse, CircuitBreaker
from ...src.query_intelligence import QueryIntelligence
from ...src.advanced_query_intelligence import AdvancedQueryIntelligence

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class QueryType(str, Enum):
    """Query classification types."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    CREATIVE = "creative"
    PROCEDURAL = "procedural"
    MULTI_STEP = "multi_step"
    CONTEXTUAL = "contextual"


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class QueryIntent(BaseModel):
    """Detected query intent and metadata."""
    primary_intent: str
    secondary_intents: List[str] = Field(default_factory=list)
    confidence: float
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    temporal_context: Optional[Dict[str, Any]] = None
    domain: Optional[str] = None


class QueryAnalysisRequest(ServiceRequest):
    """Query analysis request model."""
    query_text: str
    context: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    conversation_history: List[str] = Field(default_factory=list)
    analyze_intent: bool = True
    classify_type: bool = True
    assess_complexity: bool = True
    extract_entities: bool = True


class QueryRewriteRequest(ServiceRequest):
    """Query rewrite request model."""
    original_query: str
    context: Optional[str] = None
    rewrite_strategy: str = "enhance"  # enhance, simplify, expand, focus
    preserve_intent: bool = True
    max_alternatives: int = 3


class QueryRoutingDecision(BaseModel):
    """Query routing decision result."""
    target_service: str
    routing_confidence: float
    estimated_complexity: QueryComplexity
    recommended_strategy: str
    fallback_services: List[str] = Field(default_factory=list)
    reasoning: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EnhancedQuery(BaseModel):
    """Enhanced query with intelligence analysis."""
    original_query: str
    enhanced_query: str
    query_type: QueryType
    complexity: QueryComplexity
    intent: QueryIntent
    routing_decision: QueryRoutingDecision
    suggested_filters: Dict[str, Any] = Field(default_factory=dict)
    expansion_terms: List[str] = Field(default_factory=list)
    confidence_score: float
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryIntelligenceService(BaseService):
    """
    Enterprise query intelligence service with advanced understanding capabilities.
    Provides query analysis, enhancement, routing, and optimization.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="query-intelligence",
            version="2.0.0",
            description="Enterprise Query Intelligence and Routing Service",
            port=8002,
            **kwargs
        )
        
        self.query_intelligence = None
        self.advanced_query_intelligence = None
        self.embedding_model = None
        self.nlp_model = None
        
        # Query processing statistics
        self.query_stats = {
            "queries_analyzed": 0,
            "queries_enhanced": 0,
            "routing_decisions": 0,
            "avg_analysis_time_ms": 0.0,
            "intent_accuracy": 0.0,
            "routing_accuracy": 0.0,
            "query_types": {},
            "complexity_distribution": {}
        }
        
        # Query pattern cache
        self.pattern_cache = {}
        self.intent_cache = {}
        
        # Setup routes
        self._setup_query_routes()
    
    async def initialize(self):
        """Initialize query intelligence components."""
        try:
            self.logger.info("Initializing query intelligence service...")
            
            # Initialize core components
            self.query_intelligence = QueryIntelligence()
            self.advanced_query_intelligence = AdvancedQueryIntelligence()
            
            # Initialize embedding model for semantic analysis
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Sentence transformer model loaded")
            
            # Initialize spaCy for NLP
            if SPACY_AVAILABLE:
                try:
                    import spacy
                    self.nlp_model = spacy.load("en_core_web_sm")
                    self.logger.info("spaCy NLP model loaded")
                except OSError:
                    self.logger.warning("spaCy model not found")
            
            # Add circuit breakers
            self.add_circuit_breaker("embedding_service", failure_threshold=3, recovery_timeout=30.0)
            self.add_circuit_breaker("nlp_processing", failure_threshold=5, recovery_timeout=60.0)
            
            self.logger.info("Query intelligence service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize query intelligence service: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup query intelligence resources."""
        self.logger.info("Shutting down query intelligence service...")
        # Cleanup resources if needed
        pass
    
    async def health_check(self) -> ServiceStatus:
        """Perform health check on query intelligence components."""
        try:
            # Test basic functionality
            if not self.query_intelligence:
                return ServiceStatus.UNHEALTHY
            
            # Test a simple query analysis
            test_query = "What is artificial intelligence?"
            result = await self._analyze_query_intent(test_query)
            
            if result and result.confidence > 0.0:
                return ServiceStatus.HEALTHY
            else:
                return ServiceStatus.DEGRADED
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return ServiceStatus.UNHEALTHY
    
    def _setup_query_routes(self):
        """Setup query intelligence API routes."""
        
        @self.app.post("/api/v1/analyze", response_model=ServiceResponse)
        async def analyze_query(request: QueryAnalysisRequest):
            """Comprehensive query analysis including intent, type, and complexity."""
            start_time = datetime.utcnow()
            
            try:
                async with self.trace_operation("analyze_query"):
                    # Perform comprehensive query analysis
                    analysis_result = await self._perform_comprehensive_analysis(request)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self._update_stats("analysis", processing_time)
                    
                    return self.create_response(
                        request_id=request.request_id,
                        data=analysis_result.dict(),
                        processing_time_ms=processing_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Error analyzing query: {e}")
                self.metrics.error_count += 1
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                return self.create_response(
                    request_id=request.request_id,
                    error=f"Query analysis failed: {str(e)}",
                    processing_time_ms=processing_time
                )
        
        @self.app.post("/api/v1/enhance", response_model=ServiceResponse)
        async def enhance_query(request: QueryAnalysisRequest):
            """Enhance query with intelligent rewriting and expansion."""
            start_time = datetime.utcnow()
            
            try:
                async with self.trace_operation("enhance_query"):
                    # Enhance the query
                    enhanced_result = await self._enhance_query_intelligence(request)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self._update_stats("enhancement", processing_time)
                    
                    return self.create_response(
                        request_id=request.request_id,
                        data=enhanced_result.dict(),
                        processing_time_ms=processing_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Error enhancing query: {e}")
                self.metrics.error_count += 1
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                return self.create_response(
                    request_id=request.request_id,
                    error=f"Query enhancement failed: {str(e)}",
                    processing_time_ms=processing_time
                )
        
        @self.app.post("/api/v1/route", response_model=ServiceResponse)
        async def route_query(request: QueryAnalysisRequest):
            """Intelligent query routing to appropriate services."""
            start_time = datetime.utcnow()
            
            try:
                async with self.trace_operation("route_query"):
                    # Analyze query for routing
                    routing_decision = await self._make_routing_decision(request)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self._update_stats("routing", processing_time)
                    
                    return self.create_response(
                        request_id=request.request_id,
                        data=routing_decision.dict(),
                        processing_time_ms=processing_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Error routing query: {e}")
                self.metrics.error_count += 1
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                return self.create_response(
                    request_id=request.request_id,
                    error=f"Query routing failed: {str(e)}",
                    processing_time_ms=processing_time
                )
        
        @self.app.post("/api/v1/rewrite", response_model=ServiceResponse)
        async def rewrite_query(request: QueryRewriteRequest):
            """Intelligent query rewriting with multiple strategies."""
            start_time = datetime.utcnow()
            
            try:
                async with self.trace_operation("rewrite_query"):
                    # Rewrite query using specified strategy
                    rewritten_queries = await self._rewrite_query_with_strategy(request)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    return self.create_response(
                        request_id=request.request_id,
                        data={
                            "original_query": request.original_query,
                            "rewritten_queries": rewritten_queries,
                            "strategy": request.rewrite_strategy
                        },
                        processing_time_ms=processing_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Error rewriting query: {e}")
                self.metrics.error_count += 1
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                return self.create_response(
                    request_id=request.request_id,
                    error=f"Query rewriting failed: {str(e)}",
                    processing_time_ms=processing_time
                )
        
        @self.app.get("/api/v1/stats", response_model=ServiceResponse)
        async def get_intelligence_stats():
            """Get query intelligence service statistics."""
            try:
                stats = self.query_stats.copy()
                stats.update({
                    "service_uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                    "current_status": self.status,
                    "cache_sizes": {
                        "pattern_cache": len(self.pattern_cache),
                        "intent_cache": len(self.intent_cache)
                    },
                    "circuit_breaker_states": {
                        name: cb.state for name, cb in self.circuit_breakers.items()
                    }
                })
                
                return self.create_response(
                    request_id=str(uuid4()),
                    data=stats
                )
                
            except Exception as e:
                self.logger.error(f"Error getting stats: {e}")
                return self.create_response(
                    request_id=str(uuid4()),
                    error=f"Failed to get stats: {str(e)}"
                )
    
    async def _perform_comprehensive_analysis(self, request: QueryAnalysisRequest) -> EnhancedQuery:
        """Perform comprehensive query analysis with all intelligence features."""
        query_text = request.query_text
        
        # Check cache first
        cache_key = f"analysis:{hash(query_text)}"
        cached_result = await self.cache_get(cache_key)
        if cached_result:
            return EnhancedQuery(**cached_result)
        
        # Analyze query intent
        intent = await self._analyze_query_intent(query_text, request.context)
        
        # Classify query type
        query_type = await self._classify_query_type(query_text, intent)
        
        # Assess complexity
        complexity = await self._assess_query_complexity(query_text, intent)
        
        # Make routing decision
        routing_decision = await self._make_routing_decision(request)
        
        # Generate query enhancements
        enhanced_query = await self._generate_enhanced_query(query_text, intent, query_type)
        
        # Extract expansion terms
        expansion_terms = await self._extract_expansion_terms(query_text, intent)
        
        # Generate suggested filters
        suggested_filters = await self._generate_suggested_filters(query_text, intent)
        
        # Calculate confidence score
        confidence_score = self._calculate_overall_confidence(intent, query_type, complexity)
        
        result = EnhancedQuery(
            original_query=query_text,
            enhanced_query=enhanced_query,
            query_type=query_type,
            complexity=complexity,
            intent=intent,
            routing_decision=routing_decision,
            suggested_filters=suggested_filters,
            expansion_terms=expansion_terms,
            confidence_score=confidence_score,
            processing_metadata={
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "service_version": self.version,
                "features_used": ["intent", "classification", "complexity", "routing", "enhancement"]
            }
        )
        
        # Cache result
        await self.cache_set(cache_key, result.dict(), ttl=1800)
        
        return result
    
    async def _analyze_query_intent(self, query_text: str, context: Optional[str] = None) -> QueryIntent:
        """Analyze query intent using advanced NLP techniques."""
        try:
            # Check intent cache
            cache_key = f"intent:{hash(query_text + (context or ''))}"
            cached_intent = self.intent_cache.get(cache_key)
            if cached_intent:
                return cached_intent
            
            # Use advanced query intelligence
            intent_result = self.advanced_query_intelligence.analyze_intent(query_text, context)
            
            # Extract entities using spaCy if available
            entities = []
            if self.nlp_model:
                doc = self.nlp_model(query_text)
                entities = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "confidence": 0.9,
                        "start": ent.start_char,
                        "end": ent.end_char
                    }
                    for ent in doc.ents
                ]
            
            # Detect topics and themes
            topics = self._extract_topics(query_text, entities)
            
            # Analyze temporal context
            temporal_context = self._analyze_temporal_context(query_text)
            
            # Detect domain
            domain = self._detect_domain(query_text, entities, topics)
            
            intent = QueryIntent(
                primary_intent=intent_result.get("primary_intent", "information_retrieval"),
                secondary_intents=intent_result.get("secondary_intents", []),
                confidence=intent_result.get("confidence", 0.8),
                entities=entities,
                topics=topics,
                temporal_context=temporal_context,
                domain=domain
            )
            
            # Cache intent
            self.intent_cache[cache_key] = intent
            
            return intent
            
        except Exception as e:
            self.logger.warning(f"Intent analysis failed, using fallback: {e}")
            return QueryIntent(
                primary_intent="information_retrieval",
                confidence=0.5,
                entities=[],
                topics=[]
            )
    
    async def _classify_query_type(self, query_text: str, intent: QueryIntent) -> QueryType:
        """Classify query into semantic types."""
        try:
            # Use pattern matching and ML classification
            query_lower = query_text.lower()
            
            # Factual queries
            if any(word in query_lower for word in ["what", "who", "when", "where", "define", "definition"]):
                return QueryType.FACTUAL
            
            # Analytical queries
            elif any(word in query_lower for word in ["analyze", "explain", "why", "how does", "relationship", "impact"]):
                return QueryType.ANALYTICAL
            
            # Comparative queries
            elif any(word in query_lower for word in ["compare", "versus", "vs", "difference", "similar", "contrast"]):
                return QueryType.COMPARATIVE
            
            # Creative queries
            elif any(word in query_lower for word in ["create", "generate", "design", "suggest", "brainstorm"]):
                return QueryType.CREATIVE
            
            # Procedural queries
            elif any(word in query_lower for word in ["how to", "steps", "process", "procedure", "guide", "tutorial"]):
                return QueryType.PROCEDURAL
            
            # Multi-step queries (complex reasoning)
            elif len(query_text.split("?")) > 1 or "then" in query_lower or "after" in query_lower:
                return QueryType.MULTI_STEP
            
            else:
                return QueryType.CONTEXTUAL
                
        except Exception as e:
            self.logger.warning(f"Query type classification failed: {e}")
            return QueryType.FACTUAL
    
    async def _assess_query_complexity(self, query_text: str, intent: QueryIntent) -> QueryComplexity:
        """Assess query complexity based on multiple factors."""
        try:
            complexity_score = 0
            
            # Length-based complexity
            word_count = len(query_text.split())
            if word_count > 20:
                complexity_score += 2
            elif word_count > 10:
                complexity_score += 1
            
            # Entity complexity
            if len(intent.entities) > 3:
                complexity_score += 2
            elif len(intent.entities) > 1:
                complexity_score += 1
            
            # Question complexity
            question_count = query_text.count("?")
            if question_count > 1:
                complexity_score += 2
            elif question_count == 1:
                complexity_score += 1
            
            # Topic complexity
            if len(intent.topics) > 2:
                complexity_score += 1
            
            # Logical operators
            if any(op in query_text.lower() for op in ["and", "or", "but", "however", "because", "therefore"]):
                complexity_score += 1
            
            # Map score to complexity level
            if complexity_score >= 6:
                return QueryComplexity.EXPERT
            elif complexity_score >= 4:
                return QueryComplexity.COMPLEX
            elif complexity_score >= 2:
                return QueryComplexity.MODERATE
            else:
                return QueryComplexity.SIMPLE
                
        except Exception as e:
            self.logger.warning(f"Complexity assessment failed: {e}")
            return QueryComplexity.MODERATE
    
    async def _make_routing_decision(self, request: QueryAnalysisRequest) -> QueryRoutingDecision:
        """Make intelligent routing decision based on query analysis."""
        try:
            query_text = request.query_text
            
            # Analyze query characteristics
            has_document_context = "document" in query_text.lower() or "file" in query_text.lower()
            has_comparison = any(word in query_text.lower() for word in ["compare", "versus", "difference"])
            has_analytics = any(word in query_text.lower() for word in ["analyze", "trend", "pattern", "statistics"])
            has_generation = any(word in query_text.lower() for word in ["create", "generate", "write", "summarize"])
            
            # Route based on query characteristics
            if has_document_context:
                target_service = "document-search"
                confidence = 0.9
                strategy = "document_retrieval"
                fallbacks = ["hybrid-search", "vector-search"]
                reasoning = "Query references documents or files"
            
            elif has_comparison:
                target_service = "hybrid-search"
                confidence = 0.85
                strategy = "comparative_analysis"
                fallbacks = ["vector-search", "knowledge-graph"]
                reasoning = "Query requires comparative analysis"
            
            elif has_analytics:
                target_service = "analytics-engine"
                confidence = 0.8
                strategy = "analytical_processing"
                fallbacks = ["hybrid-search", "document-search"]
                reasoning = "Query requires analytical processing"
            
            elif has_generation:
                target_service = "generation-service"
                confidence = 0.9
                strategy = "generative_response"
                fallbacks = ["hybrid-search"]
                reasoning = "Query requires content generation"
            
            else:
                target_service = "hybrid-search"
                confidence = 0.75
                strategy = "general_retrieval"
                fallbacks = ["vector-search", "document-search"]
                reasoning = "General information retrieval query"
            
            # Assess complexity for routing
            complexity = await self._assess_query_complexity(query_text, QueryIntent(
                primary_intent="unknown", confidence=0.5, entities=[], topics=[]
            ))
            
            return QueryRoutingDecision(
                target_service=target_service,
                routing_confidence=confidence,
                estimated_complexity=complexity,
                recommended_strategy=strategy,
                fallback_services=fallbacks,
                reasoning=reasoning,
                metadata={
                    "routing_timestamp": datetime.utcnow().isoformat(),
                    "query_characteristics": {
                        "has_document_context": has_document_context,
                        "has_comparison": has_comparison,
                        "has_analytics": has_analytics,
                        "has_generation": has_generation
                    }
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Routing decision failed, using default: {e}")
            return QueryRoutingDecision(
                target_service="hybrid-search",
                routing_confidence=0.5,
                estimated_complexity=QueryComplexity.MODERATE,
                recommended_strategy="general_retrieval",
                reasoning="Default routing due to analysis failure"
            )
    
    def _extract_topics(self, query_text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Extract topics and themes from query text."""
        topics = []
        
        # Domain-specific keywords
        domain_keywords = {
            "technology": ["AI", "machine learning", "software", "algorithm", "data", "system"],
            "business": ["revenue", "profit", "market", "strategy", "customer", "sales"],
            "science": ["research", "study", "experiment", "analysis", "hypothesis", "theory"],
            "healthcare": ["medical", "health", "patient", "treatment", "diagnosis", "therapy"],
            "finance": ["investment", "portfolio", "risk", "return", "trading", "valuation"]
        }
        
        query_lower = query_text.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword.lower() in query_lower for keyword in keywords):
                topics.append(domain)
        
        # Add entity labels as topics
        for entity in entities:
            if entity["label"] not in ["PERSON", "ORG", "GPE"]:  # Skip basic entity types
                topics.append(entity["label"].lower())
        
        return list(set(topics))
    
    def _analyze_temporal_context(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Analyze temporal context in query."""
        temporal_patterns = {
            "recent": ["recent", "latest", "current", "now", "today"],
            "historical": ["historical", "past", "previous", "before", "earlier"],
            "future": ["future", "upcoming", "next", "will be", "projected"],
            "specific": r"\b(19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
        }
        
        query_lower = query_text.lower()
        temporal_context = {}
        
        for context_type, patterns in temporal_patterns.items():
            if context_type == "specific":
                if re.search(patterns, query_text):
                    temporal_context["type"] = "specific"
                    temporal_context["patterns"] = re.findall(patterns, query_text)
            else:
                if any(pattern in query_lower for pattern in patterns):
                    temporal_context["type"] = context_type
                    temporal_context["indicators"] = [p for p in patterns if p in query_lower]
        
        return temporal_context if temporal_context else None
    
    def _detect_domain(self, query_text: str, entities: List[Dict[str, Any]], topics: List[str]) -> Optional[str]:
        """Detect the primary domain of the query."""
        domain_indicators = {
            "technology": ["software", "AI", "algorithm", "system", "tech", "programming"],
            "business": ["business", "market", "revenue", "customer", "strategy"],
            "science": ["research", "study", "scientific", "experiment", "data"],
            "healthcare": ["medical", "health", "patient", "clinical", "treatment"],
            "legal": ["legal", "law", "contract", "regulation", "compliance"],
            "finance": ["financial", "investment", "trading", "portfolio", "risk"]
        }
        
        query_lower = query_text.lower()
        domain_scores = {}
        
        # Score based on keywords
        for domain, keywords in domain_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Boost score based on topics
        for topic in topics:
            for domain, keywords in domain_indicators.items():
                if topic in keywords:
                    domain_scores[domain] = domain_scores.get(domain, 0) + 2
        
        # Return highest scoring domain
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    async def _enhance_query_intelligence(self, request: QueryAnalysisRequest) -> EnhancedQuery:
        """Enhance query with intelligent analysis and improvements."""
        # This would use the comprehensive analysis
        return await self._perform_comprehensive_analysis(request)
    
    async def _generate_enhanced_query(self, query_text: str, intent: QueryIntent, query_type: QueryType) -> str:
        """Generate an enhanced version of the query."""
        try:
            # Use the advanced query intelligence for enhancement
            enhanced = self.advanced_query_intelligence.enhance_query(query_text, intent.dict())
            return enhanced.get("enhanced_query", query_text)
        except Exception as e:
            self.logger.warning(f"Query enhancement failed: {e}")
            return query_text
    
    async def _extract_expansion_terms(self, query_text: str, intent: QueryIntent) -> List[str]:
        """Extract terms that could expand the query scope."""
        expansion_terms = []
        
        # Use entity information for expansion
        for entity in intent.entities:
            if entity["label"] in ["ORG", "PRODUCT", "TECHNOLOGY"]:
                expansion_terms.append(entity["text"])
        
        # Domain-specific expansions
        if intent.domain:
            domain_expansions = {
                "technology": ["software", "system", "platform", "solution"],
                "business": ["strategy", "process", "management", "operations"],
                "science": ["methodology", "findings", "research", "analysis"]
            }
            expansion_terms.extend(domain_expansions.get(intent.domain, []))
        
        return list(set(expansion_terms))
    
    async def _generate_suggested_filters(self, query_text: str, intent: QueryIntent) -> Dict[str, Any]:
        """Generate suggested filters based on query analysis."""
        filters = {}
        
        # Temporal filters
        if intent.temporal_context:
            filters["temporal"] = intent.temporal_context
        
        # Domain filters
        if intent.domain:
            filters["domain"] = [intent.domain]
        
        # Entity filters
        if intent.entities:
            filters["entities"] = [ent["text"] for ent in intent.entities]
        
        # Topic filters
        if intent.topics:
            filters["topics"] = intent.topics
        
        return filters
    
    async def _rewrite_query_with_strategy(self, request: QueryRewriteRequest) -> List[str]:
        """Rewrite query using specified strategy."""
        try:
            original = request.original_query
            strategy = request.rewrite_strategy
            max_alternatives = request.max_alternatives
            
            alternatives = []
            
            if strategy == "enhance":
                # Make query more specific and detailed
                enhanced = f"Please provide detailed information about {original.lower()}"
                alternatives.append(enhanced)
                
                # Add context-aware enhancement
                if request.context:
                    contextual = f"In the context of {request.context}, {original.lower()}"
                    alternatives.append(contextual)
            
            elif strategy == "simplify":
                # Simplify complex queries
                simplified = re.sub(r'\b(please|could you|would you)\b', '', original, flags=re.IGNORECASE)
                simplified = simplified.strip()
                alternatives.append(simplified)
            
            elif strategy == "expand":
                # Expand query scope
                expanded = f"{original} including related concepts, examples, and applications"
                alternatives.append(expanded)
            
            elif strategy == "focus":
                # Focus on specific aspects
                focused = f"Specifically regarding {original.lower()}, what are the key points?"
                alternatives.append(focused)
            
            # Limit alternatives
            return alternatives[:max_alternatives]
            
        except Exception as e:
            self.logger.warning(f"Query rewriting failed: {e}")
            return [request.original_query]
    
    def _calculate_overall_confidence(self, intent: QueryIntent, query_type: QueryType, complexity: QueryComplexity) -> float:
        """Calculate overall confidence score for the analysis."""
        confidence_factors = [
            intent.confidence,  # Intent confidence
            0.9 if query_type != QueryType.CONTEXTUAL else 0.7,  # Type classification confidence
            0.9 if complexity != QueryComplexity.EXPERT else 0.8,  # Complexity assessment confidence
        ]
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _update_stats(self, operation_type: str, processing_time: float):
        """Update service statistics."""
        if operation_type == "analysis":
            self.query_stats["queries_analyzed"] += 1
        elif operation_type == "enhancement":
            self.query_stats["queries_enhanced"] += 1
        elif operation_type == "routing":
            self.query_stats["routing_decisions"] += 1
        
        # Update average processing time
        total_ops = sum([
            self.query_stats["queries_analyzed"],
            self.query_stats["queries_enhanced"],
            self.query_stats["routing_decisions"]
        ])
        
        if total_ops > 0:
            prev_avg = self.query_stats["avg_analysis_time_ms"]
            self.query_stats["avg_analysis_time_ms"] = (
                (prev_avg * (total_ops - 1) + processing_time) / total_ops
            )


# Entry point for running the service
if __name__ == "__main__":
    service = QueryIntelligenceService()
    service.run(debug=True)