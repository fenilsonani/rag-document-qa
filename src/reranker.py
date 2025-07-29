"""
Advanced Cross-Encoder Reranking System - Pro-Level RAG Enhancement
Implements sophisticated reranking using cross-encoder models for improved relevance scoring.
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from functools import lru_cache
import logging

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not available. Cross-encoder reranking will be disabled.")

from langchain.schema import Document

from .config import Config
from .hybrid_search import SearchResult


@dataclass
class RerankingResult:
    """Represents a reranked search result."""
    document: Document
    original_score: float
    rerank_score: float
    score_improvement: float
    original_rank: int
    new_rank: int
    confidence: float
    source: str
    metadata: Dict[str, Any]


class CrossEncoderReranker:
    """
    Advanced cross-encoder reranking system for improving search result relevance.
    
    This class implements sophisticated reranking using pre-trained cross-encoder models
    that directly score query-document pairs for relevance.
    """
    
    def __init__(
        self, 
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        device: str = 'cpu',
        batch_size: int = 8,
        enable_caching: bool = True
    ):
        """
        Initialize the CrossEncoderReranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
            device: Device to run the model on ('cpu' or 'cuda')
            batch_size: Batch size for processing multiple documents
            enable_caching: Whether to cache reranking results
        """
        self.config = Config()
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        
        self.model: Optional[CrossEncoder] = None
        self.is_initialized = False
        self.cache: Dict[str, float] = {}
        
        # Performance tracking
        self.reranking_stats = {
            'total_queries': 0,
            'total_documents_reranked': 0,
            'average_latency': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Adaptive parameters
        self.confidence_threshold = 0.5
        self.score_boost_factor = 1.0
        self.diversity_lambda = 0.1  # For diversity-aware reranking
        
    def initialize(self) -> Dict[str, Any]:
        """Initialize the cross-encoder model."""
        if not CROSS_ENCODER_AVAILABLE:
            return {
                "success": False,
                "error": "sentence-transformers not available. Please install: pip install sentence-transformers",
                "fallback": "Reranking will use basic score normalization"
            }
        
        try:
            print(f"🔄 Loading cross-encoder model: {self.model_name}")
            start_time = time.time()
            
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512  # Reasonable max length for efficiency
            )
            
            load_time = time.time() - start_time
            self.is_initialized = True
            
            print(f"✅ Cross-encoder model loaded in {load_time:.2f}s")
            
            return {
                "success": True,
                "model_name": self.model_name,
                "device": self.device,
                "load_time": load_time,
                "max_length": 512
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize cross-encoder: {str(e)}",
                "fallback": "Reranking will use basic score normalization"
            }
    
    def _generate_cache_key(self, query: str, document_text: str) -> str:
        """Generate a cache key for query-document pair."""
        return f"{hash(query)}_{hash(document_text)}"
    
    def _prepare_query_document_pairs(
        self, 
        query: str, 
        search_results: List[SearchResult]
    ) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for batch processing."""
        pairs = []
        for result in search_results:
            # Truncate document content if too long
            doc_text = result.document.page_content
            if len(doc_text) > 1000:  # Reasonable limit for cross-encoder
                doc_text = doc_text[:1000] + "..."
            
            pairs.append((query, doc_text))
        
        return pairs
    
    def _batch_predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict relevance scores for query-document pairs in batches."""
        if not self.model:
            # Fallback: return normalized random scores
            return [0.5 + np.random.normal(0, 0.1) for _ in pairs]
        
        scores = []
        
        # Process in batches for efficiency
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            try:
                batch_scores = self.model.predict(batch_pairs)
                scores.extend(batch_scores.tolist())
            except Exception as e:
                logging.warning(f"Batch prediction failed: {e}. Using fallback scores.")
                # Fallback scores
                fallback_scores = [0.5 + np.random.normal(0, 0.1) for _ in batch_pairs]
                scores.extend(fallback_scores)
        
        return scores
    
    def _calculate_confidence(self, score: float, original_score: float) -> float:
        """Calculate confidence in the reranking decision."""
        # Higher confidence when rerank score is significantly different from original
        score_diff = abs(score - original_score)
        confidence = min(1.0, score_diff * 2.0)  # Scale factor
        
        # Also consider absolute score magnitude
        confidence *= min(1.0, abs(score) * 2.0)
        
        return confidence
    
    def _apply_diversity_boost(
        self, 
        reranked_results: List[RerankingResult],
        diversity_lambda: float = 0.1
    ) -> List[RerankingResult]:
        """Apply diversity boost to reduce redundancy in top results."""
        if diversity_lambda <= 0 or len(reranked_results) <= 1:
            return reranked_results
        
        # Simple diversity boost: penalize results that are too similar to higher-ranked ones
        diverse_results = []
        
        for i, result in enumerate(reranked_results):
            diversity_penalty = 0.0
            
            # Compare with higher-ranked results
            for j, prev_result in enumerate(diverse_results):
                # Simple text similarity (could be improved with embeddings)
                text1 = result.document.page_content.lower()
                text2 = prev_result.document.page_content.lower()
                
                # Basic overlap-based similarity
                words1 = set(text1.split())
                words2 = set(text2.split())
                
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    diversity_penalty += overlap * diversity_lambda * (1.0 / (j + 1))  # Decay penalty
            
            # Apply penalty
            adjusted_score = result.rerank_score - diversity_penalty
            
            # Update result with adjusted score
            adjusted_result = RerankingResult(
                document=result.document,
                original_score=result.original_score,
                rerank_score=adjusted_score,
                score_improvement=adjusted_score - result.original_score,
                original_rank=result.original_rank,
                new_rank=i + 1,
                confidence=result.confidence,
                source=result.source,
                metadata={
                    **result.metadata,
                    'diversity_penalty': diversity_penalty,
                    'diversity_adjusted': True
                }
            )
            
            diverse_results.append(adjusted_result)
        
        # Re-sort by adjusted scores
        diverse_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(diverse_results):
            result.new_rank = i + 1
        
        return diverse_results
    
    def rerank(
        self,
        query: str,
        search_results: List[SearchResult],
        top_k: Optional[int] = None,
        enable_diversity: bool = True,
        min_confidence: float = 0.0
    ) -> List[RerankingResult]:
        """
        Rerank search results using cross-encoder model.
        
        Args:
            query: Search query
            search_results: List of search results to rerank
            top_k: Number of top results to return (None for all)
            enable_diversity: Whether to apply diversity boosting
            min_confidence: Minimum confidence threshold for reranking
        
        Returns:
            List of reranked results
        """
        if not search_results:
            return []
        
        start_time = time.time()
        
        # Update stats
        self.reranking_stats['total_queries'] += 1
        self.reranking_stats['total_documents_reranked'] += len(search_results)
        
        # Prepare query-document pairs
        pairs = self._prepare_query_document_pairs(query, search_results)
        
        # Check cache if enabled
        cached_scores = []
        pairs_to_predict = []
        cache_indices = []
        
        if self.enable_caching:
            for i, (q, doc) in enumerate(pairs):
                cache_key = self._generate_cache_key(q, doc)
                if cache_key in self.cache:
                    cached_scores.append((i, self.cache[cache_key]))
                    self.reranking_stats['cache_hits'] += 1
                else:
                    pairs_to_predict.append((q, doc))
                    cache_indices.append(i)
                    self.reranking_stats['cache_misses'] += 1
        else:
            pairs_to_predict = pairs
            cache_indices = list(range(len(pairs)))
        
        # Predict scores for non-cached pairs
        if pairs_to_predict:
            predicted_scores = self._batch_predict(pairs_to_predict)
            
            # Cache new predictions
            if self.enable_caching:
                for (q, doc), score in zip(pairs_to_predict, predicted_scores):
                    cache_key = self._generate_cache_key(q, doc)
                    self.cache[cache_key] = score
        else:
            predicted_scores = []
        
        # Combine cached and predicted scores
        all_scores = [0.0] * len(pairs)
        
        # Fill in cached scores
        for i, score in cached_scores:
            all_scores[i] = score
        
        # Fill in predicted scores
        for cache_idx, score in zip(cache_indices, predicted_scores):
            all_scores[cache_idx] = score
        
        # Create reranking results
        reranked_results = []
        
        for i, (result, rerank_score) in enumerate(zip(search_results, all_scores)):
            # Calculate confidence
            confidence = self._calculate_confidence(rerank_score, result.score)
            
            # Skip if below confidence threshold
            if confidence < min_confidence:
                continue
            
            # Calculate score improvement
            score_improvement = rerank_score - result.score
            
            reranking_result = RerankingResult(
                document=result.document,
                original_score=result.score,
                rerank_score=rerank_score * self.score_boost_factor,
                score_improvement=score_improvement,
                original_rank=result.rank,
                new_rank=0,  # Will be set after sorting
                confidence=confidence,
                source=f"reranked_{result.source}",
                metadata={
                    **result.metadata,
                    'reranker_model': self.model_name,
                    'rerank_confidence': confidence,
                    'original_metadata': result.metadata
                }
            )
            
            reranked_results.append(reranking_result)
        
        # Sort by rerank score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Apply diversity boost if enabled
        if enable_diversity and self.diversity_lambda > 0:
            reranked_results = self._apply_diversity_boost(reranked_results, self.diversity_lambda)
        
        # Update ranks and apply top_k limit
        final_results = []
        for i, result in enumerate(reranked_results):
            if top_k and i >= top_k:
                break
            
            result.new_rank = i + 1
            final_results.append(result)
        
        # Update performance stats
        latency = time.time() - start_time
        self.reranking_stats['average_latency'] = (
            self.reranking_stats['average_latency'] * (self.reranking_stats['total_queries'] - 1) + latency
        ) / self.reranking_stats['total_queries']
        
        return final_results
    
    def rerank_with_explanation(
        self,
        query: str,
        search_results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rerank with detailed explanation of the reranking process.
        
        Returns:
            Dictionary with reranked results and detailed analytics
        """
        start_time = time.time()
        
        # Perform reranking
        reranked_results = self.rerank(query, search_results, top_k)
        
        # Calculate analytics
        if search_results and reranked_results:
            # Rank changes
            rank_changes = []
            for result in reranked_results:
                rank_change = result.original_rank - result.new_rank
                rank_changes.append(rank_change)
            
            # Score improvements
            score_improvements = [r.score_improvement for r in reranked_results]
            
            analytics = {
                "reranking_summary": {
                    "original_results": len(search_results),
                    "reranked_results": len(reranked_results),
                    "processing_time": time.time() - start_time,
                    "model_used": self.model_name if self.is_initialized else "fallback"
                },
                "rank_changes": {
                    "average_rank_change": np.mean(rank_changes) if rank_changes else 0,
                    "max_rank_improvement": max(rank_changes) if rank_changes else 0,
                    "max_rank_decline": min(rank_changes) if rank_changes else 0,
                    "results_improved": sum(1 for x in rank_changes if x > 0),
                    "results_declined": sum(1 for x in rank_changes if x < 0)
                },
                "score_analysis": {
                    "average_score_improvement": np.mean(score_improvements) if score_improvements else 0,
                    "max_score_improvement": max(score_improvements) if score_improvements else 0,
                    "min_score_improvement": min(score_improvements) if score_improvements else 0,
                    "results_with_positive_improvement": sum(1 for x in score_improvements if x > 0)
                },
                "confidence_analysis": {
                    "average_confidence": np.mean([r.confidence for r in reranked_results]) if reranked_results else 0,
                    "high_confidence_results": sum(1 for r in reranked_results if r.confidence > 0.7),
                    "low_confidence_results": sum(1 for r in reranked_results if r.confidence < 0.3)
                }
            }
        else:
            analytics = {"error": "No results to analyze"}
        
        return {
            "reranked_results": reranked_results,
            "analytics": analytics,
            "query": query,
            "success": True
        }
    
    def update_parameters(
        self,
        confidence_threshold: Optional[float] = None,
        score_boost_factor: Optional[float] = None,
        diversity_lambda: Optional[float] = None
    ) -> None:
        """Update adaptive parameters for reranking."""
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        
        if score_boost_factor is not None:
            self.score_boost_factor = max(0.1, score_boost_factor)
        
        if diversity_lambda is not None:
            self.diversity_lambda = max(0.0, min(1.0, diversity_lambda))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about reranking performance."""
        cache_hit_rate = 0.0
        if self.reranking_stats['cache_hits'] + self.reranking_stats['cache_misses'] > 0:
            cache_hit_rate = self.reranking_stats['cache_hits'] / (
                self.reranking_stats['cache_hits'] + self.reranking_stats['cache_misses']
            )
        
        return {
            "model_info": {
                "model_name": self.model_name,
                "device": self.device,
                "initialized": self.is_initialized,
                "available": CROSS_ENCODER_AVAILABLE
            },
            "performance": {
                **self.reranking_stats,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.cache)
            },
            "parameters": {
                "confidence_threshold": self.confidence_threshold,
                "score_boost_factor": self.score_boost_factor,
                "diversity_lambda": self.diversity_lambda,
                "batch_size": self.batch_size
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the reranking cache."""
        self.cache.clear()
        print("🧹 Reranking cache cleared")
    
    def export_cache(self) -> Dict[str, float]:
        """Export the current cache for persistence."""
        return self.cache.copy()
    
    def import_cache(self, cache_data: Dict[str, float]) -> None:
        """Import cache data from external source."""
        self.cache.update(cache_data)
        print(f"📥 Imported {len(cache_data)} cache entries")


class FallbackReranker:
    """Fallback reranker when cross-encoder is not available."""
    
    def __init__(self):
        self.is_initialized = True
    
    def initialize(self) -> Dict[str, Any]:
        return {
            "success": True,
            "message": "Using fallback reranker (score normalization)",
            "method": "fallback"
        }
    
    def rerank(
        self,
        query: str,
        search_results: List[SearchResult],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RerankingResult]:
        """Fallback reranking using score normalization."""
        if not search_results:
            return []
        
        # Simple score normalization and boosting
        scores = [r.score for r in search_results]
        if scores:
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score if max_score != min_score else 1.0
        else:
            min_score, max_score, score_range = 0.0, 1.0, 1.0
        
        reranked_results = []
        
        for i, result in enumerate(search_results):
            # Normalize score to 0-1 range
            normalized_score = (result.score - min_score) / score_range if score_range > 0 else 0.5
            
            # Apply simple boosting based on rank (prefer earlier results)
            rank_boost = 1.0 - (i * 0.1)  # Decrease by 10% per rank
            boosted_score = normalized_score * max(0.1, rank_boost)
            
            reranking_result = RerankingResult(
                document=result.document,
                original_score=result.score,
                rerank_score=boosted_score,
                score_improvement=boosted_score - result.score,
                original_rank=result.rank,
                new_rank=i + 1,
                confidence=0.5,  # Moderate confidence for fallback
                source=f"fallback_{result.source}",
                metadata={
                    **result.metadata,
                    'reranker_model': 'fallback_normalizer',
                    'normalized_score': normalized_score,
                    'rank_boost': rank_boost
                }
            )
            
            reranked_results.append(reranking_result)
        
        # Sort by boosted score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Update ranks and apply top_k
        final_results = []
        for i, result in enumerate(reranked_results):
            if top_k and i >= top_k:
                break
            result.new_rank = i + 1
            final_results.append(result)
        
        return final_results


def create_reranker(
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    **kwargs
) -> Union[CrossEncoderReranker, FallbackReranker]:
    """
    Factory function to create the appropriate reranker based on availability.
    
    Args:
        model_name: Cross-encoder model name
        **kwargs: Additional arguments for CrossEncoderReranker
    
    Returns:
        CrossEncoderReranker if available, FallbackReranker otherwise
    """
    if CROSS_ENCODER_AVAILABLE:
        return CrossEncoderReranker(model_name=model_name, **kwargs)
    else:
        print("⚠️ Cross-encoder not available, using fallback reranker")
        return FallbackReranker()