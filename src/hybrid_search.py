"""
Advanced Hybrid Search System - Pro-Level RAG Enhancement
Combines BM25 (lexical) and dense vector search with Reciprocal Rank Fusion.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from functools import lru_cache

from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .vector_store import VectorStoreManager


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    document: Document
    score: float
    rank: int
    source: str  # 'bm25', 'vector', 'hybrid'
    metadata: Dict[str, Any]


class BM25Retriever:
    """Advanced BM25 implementation for lexical search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Field length normalization parameter
        self.documents: List[Document] = []
        self.document_frequencies: Dict[str, int] = {}
        self.document_lengths: List[int] = []
        self.avg_document_length: float = 0.0
        self.vocabulary: set[str] = set()
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        self.is_fitted: bool = False
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 calculation."""
        # Convert to lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        
        # Remove common stop words (basic set for performance)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        return [token for token in tokens if token not in stop_words and len(token) > 2]
    
    def fit(self, documents: List[Document]) -> None:
        """Fit the BM25 model on documents."""
        self.documents = documents
        self.document_lengths = []
        self.document_frequencies = Counter()
        self.vocabulary = set()
        self.inverted_index = defaultdict(list)
        
        # Process each document
        for doc_idx, document in enumerate(documents):
            tokens = self._preprocess_text(document.page_content)
            self.document_lengths.append(len(tokens))
            
            # Track unique terms in this document
            unique_tokens = set(tokens)
            self.vocabulary.update(unique_tokens)
            
            # Update document frequency for each unique term
            for token in unique_tokens:
                self.document_frequencies[token] += 1
                self.inverted_index[token].append(doc_idx)
        
        # Calculate average document length
        self.avg_document_length = sum(self.document_lengths) / len(self.document_lengths)
        self.is_fitted = True
        
    def _calculate_idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency for a term."""
        if term not in self.document_frequencies:
            return 0.0
        
        df = self.document_frequencies[term]
        N = len(self.documents)
        
        # Standard IDF calculation with smoothing
        return math.log((N - df + 0.5) / (df + 0.5))
    
    def _calculate_score(self, query_terms: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document given query terms."""
        if doc_idx >= len(self.documents):
            return 0.0
            
        document_tokens = self._preprocess_text(self.documents[doc_idx].page_content)
        term_frequencies = Counter(document_tokens)
        doc_length = self.document_lengths[doc_idx]
        
        score = 0.0
        
        for term in query_terms:
            if term not in term_frequencies:
                continue
                
            tf = term_frequencies[term]
            idf = self._calculate_idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_document_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using BM25 algorithm."""
        if not self.is_fitted:
            raise ValueError("BM25 model not fitted. Call fit() first.")
        
        query_terms = self._preprocess_text(query)
        if not query_terms:
            return []
        
        # Get candidate documents (documents containing at least one query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
        
        # Score all candidate documents
        scored_docs = []
        for doc_idx in candidate_docs:
            score = self._calculate_score(query_terms, doc_idx)
            if score > 0:
                scored_docs.append((doc_idx, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_idx, score) in enumerate(scored_docs[:k]):
            results.append(SearchResult(
                document=self.documents[doc_idx],
                score=score,
                rank=rank + 1,
                source='bm25',
                metadata={
                    'bm25_score': score,
                    'doc_index': doc_idx,
                    'query_terms_matched': len([t for t in query_terms if t in self._preprocess_text(self.documents[doc_idx].page_content)])
                }
            ))
        
        return results


class QueryExpander:
    """Expand queries using synonym detection and term analysis."""
    
    def __init__(self):
        self.synonym_cache: Dict[str, List[str]] = {}
        
    @lru_cache(maxsize=1000)
    def expand_query(self, query: str, documents: List[Document] = None) -> str:
        """Expand query with related terms and synonyms."""
        # Basic query expansion using word variations
        expanded_terms = []
        terms = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        
        for term in terms:
            expanded_terms.append(term)
            
            # Add common variations
            if term.endswith('s') and len(term) > 3:
                expanded_terms.append(term[:-1])  # Remove plural
            elif not term.endswith('s'):
                expanded_terms.append(term + 's')  # Add plural
                
            # Add stemmed versions for common suffixes
            if term.endswith('ing') and len(term) > 5:
                expanded_terms.append(term[:-3])
            elif term.endswith('ed') and len(term) > 4:
                expanded_terms.append(term[:-2])
        
        # Create expanded query (remove duplicates but preserve order)
        unique_terms = []
        seen = set()
        for term in expanded_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        return ' '.join(unique_terms)


class ReciprocalRankFusion:
    """Implements Reciprocal Rank Fusion for combining multiple ranking lists."""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF with parameter k.
        
        Args:
            k: RRF parameter, typically 60. Higher values give less weight to top-ranked items.
        """
        self.k = k
    
    def fuse_rankings(
        self, 
        rankings: List[List[SearchResult]], 
        weights: Optional[List[float]] = None
    ) -> List[SearchResult]:
        """
        Fuse multiple ranking lists using RRF.
        
        Args:
            rankings: List of ranking lists to fuse
            weights: Optional weights for each ranking list
        
        Returns:
            Fused ranking list
        """
        if not rankings:
            return []
        
        if weights is None:
            weights = [1.0] * len(rankings)
        
        if len(weights) != len(rankings):
            raise ValueError("Number of weights must match number of rankings")
        
        # Create a mapping from document to its scores across rankings
        document_scores: Dict[str, float] = defaultdict(float)
        document_info: Dict[str, SearchResult] = {}
        
        for ranking_idx, ranking in enumerate(rankings):
            weight = weights[ranking_idx]
            
            for rank, result in enumerate(ranking):
                # Use document content hash as unique identifier
                doc_id = hash(result.document.page_content)
                
                # RRF formula: score = weight / (k + rank)
                rrf_score = weight / (self.k + rank + 1)  # rank is 0-indexed
                document_scores[doc_id] += rrf_score
                
                # Store document info (use latest encounter)
                if doc_id not in document_info:
                    document_info[doc_id] = result
        
        # Sort by fused scores
        sorted_docs = sorted(
            document_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Create final ranking
        fused_results = []
        for new_rank, (doc_id, fused_score) in enumerate(sorted_docs):
            original_result = document_info[doc_id]
            
            # Create new result with fused score
            fused_result = SearchResult(
                document=original_result.document,
                score=fused_score,
                rank=new_rank + 1,
                source='hybrid',
                metadata={
                    **original_result.metadata,
                    'rrf_score': fused_score,
                    'original_source': original_result.source,
                    'original_score': original_result.score,
                    'original_rank': original_result.rank
                }
            )
            
            fused_results.append(fused_result)
        
        return fused_results


class HybridSearchEngine:
    """Advanced hybrid search combining BM25 and dense vector search."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, enable_reranking: bool = True):
        self.config = Config()
        self.vector_store_manager = vector_store_manager
        self.bm25_retriever = BM25Retriever()
        self.query_expander = QueryExpander()
        self.rrf = ReciprocalRankFusion()
        self.is_initialized = False
        
        # Reranking support
        self.enable_reranking = enable_reranking
        self.reranker = None
        
        # Configurable parameters
        self.bm25_weight = 0.5
        self.vector_weight = 0.5
        self.enable_query_expansion = True
        
    def initialize(self, documents: List[Document]) -> Dict[str, Any]:
        """Initialize the hybrid search engine with documents."""
        try:
            # Fit BM25 on documents
            self.bm25_retriever.fit(documents)
            
            # Ensure vector store is ready
            if not self.vector_store_manager.vector_store:
                self.vector_store_manager.create_vector_store(documents)
            
            # Initialize reranker if enabled
            reranker_status = {"enabled": False}
            if self.enable_reranking:
                try:
                    from .reranker import create_reranker
                    self.reranker = create_reranker()
                    reranker_init_result = self.reranker.initialize()
                    reranker_status = {
                        "enabled": True,
                        "success": reranker_init_result["success"],
                        "model": reranker_init_result.get("model_name", "fallback")
                    }
                except ImportError as e:
                    print(f"⚠️ Reranker import failed: {e}")
                    reranker_status = {"enabled": False, "error": "Import failed"}
                except Exception as e:
                    print(f"⚠️ Reranker initialization failed: {e}")
                    reranker_status = {"enabled": False, "error": str(e)}
            
            self.is_initialized = True
            
            return {
                "success": True,
                "message": "Hybrid search engine initialized",
                "document_count": len(documents),
                "bm25_vocabulary_size": len(self.bm25_retriever.vocabulary),
                "vector_store_ready": self.vector_store_manager.vector_store is not None,
                "reranker_status": reranker_status
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize hybrid search: {str(e)}"
            }
    
    def search_bm25(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using BM25 only."""
        if not self.is_initialized:
            raise ValueError("Hybrid search engine not initialized")
        
        # Optionally expand query
        if self.enable_query_expansion:
            expanded_query = self.query_expander.expand_query(query)
        else:
            expanded_query = query
        
        return self.bm25_retriever.search(expanded_query, k)
    
    def search_vector(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using dense vector similarity only."""
        if not self.is_initialized:
            raise ValueError("Hybrid search engine not initialized")
        
        try:
            # Get results with scores
            results_with_scores = self.vector_store_manager.similarity_search_with_score(
                query=query, 
                k=k
            )
            
            # Convert to SearchResult format
            search_results = []
            for rank, (document, score) in enumerate(results_with_scores):
                search_results.append(SearchResult(
                    document=document,
                    score=float(score),  # ChromaDB returns distances, lower is better
                    rank=rank + 1,
                    source='vector',
                    metadata={
                        'vector_score': float(score),
                        'similarity_score': 1.0 / (1.0 + score)  # Convert distance to similarity
                    }
                ))
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"Vector search failed: {str(e)}")
    
    def search_hybrid(
        self, 
        query: str, 
        k: int = 10,
        bm25_k: Optional[int] = None,
        vector_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            k: Final number of results to return
            bm25_k: Number of results to get from BM25 (default: k * 2)
            vector_k: Number of results to get from vector search (default: k * 2)
        """
        if not self.is_initialized:
            raise ValueError("Hybrid search engine not initialized")
        
        # Get more results from each method to improve fusion
        bm25_k = bm25_k or min(k * 2, 20)
        vector_k = vector_k or min(k * 2, 20)
        
        # Perform both searches
        bm25_results = self.search_bm25(query, bm25_k)
        vector_results = self.search_vector(query, vector_k)
        
        # Fuse results using RRF
        fused_results = self.rrf.fuse_rankings(
            rankings=[bm25_results, vector_results],
            weights=[self.bm25_weight, self.vector_weight]
        )
        
        # Return top k results
        return fused_results[:k]
    
    def search(
        self, 
        query: str, 
        k: int = 10, 
        method: str = 'hybrid',
        enable_reranking: bool = True,
        rerank_top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Main search interface with optional reranking.
        
        Args:
            query: Search query
            k: Number of results to return
            method: Search method ('bm25', 'vector', 'hybrid')
            enable_reranking: Whether to apply reranking
            rerank_top_k: Number of top results to rerank (default: k * 2)
        """
        # Get initial results
        if method == 'bm25':
            results = self.search_bm25(query, k * 2 if enable_reranking else k)
        elif method == 'vector':
            results = self.search_vector(query, k * 2 if enable_reranking else k)
        elif method == 'hybrid':
            results = self.search_hybrid(query, k * 2 if enable_reranking else k)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        # Apply reranking if enabled and available
        if (enable_reranking and 
            self.enable_reranking and 
            self.reranker and 
            len(results) > 1):
            
            try:
                from .reranker import RerankingResult
                
                # Rerank results
                reranked = self.reranker.rerank(
                    query=query,
                    search_results=results,
                    top_k=rerank_top_k or k
                )
                
                # Convert back to SearchResult format
                reranked_search_results = []
                for rerank_result in reranked[:k]:
                    search_result = SearchResult(
                        document=rerank_result.document,
                        score=rerank_result.rerank_score,
                        rank=rerank_result.new_rank,
                        source=f"reranked_{method}",
                        metadata={
                            **rerank_result.metadata,
                            'original_score': rerank_result.original_score,
                            'original_rank': rerank_result.original_rank,
                            'rerank_confidence': rerank_result.confidence,
                            'score_improvement': rerank_result.score_improvement
                        }
                    )
                    reranked_search_results.append(search_result)
                
                return reranked_search_results
                
            except Exception as e:
                print(f"⚠️ Reranking failed: {e}, returning original results")
                return results[:k]
        
        return results[:k]
    
    def get_search_analytics(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Get detailed analytics for a search query."""
        if not self.is_initialized:
            raise ValueError("Hybrid search engine not initialized")
        
        # Perform all three searches
        bm25_results = self.search_bm25(query, k)
        vector_results = self.search_vector(query, k)
        hybrid_results = self.search_hybrid(query, k)
        
        # Calculate overlap metrics
        bm25_docs = {hash(r.document.page_content) for r in bm25_results}
        vector_docs = {hash(r.document.page_content) for r in vector_results}
        hybrid_docs = {hash(r.document.page_content) for r in hybrid_results}
        
        overlap_bm25_vector = len(bm25_docs & vector_docs)
        overlap_all = len(bm25_docs & vector_docs & hybrid_docs)
        
        return {
            "query": query,
            "results_count": {
                "bm25": len(bm25_results),
                "vector": len(vector_results),
                "hybrid": len(hybrid_results)
            },
            "overlap_metrics": {
                "bm25_vector_overlap": overlap_bm25_vector,
                "all_methods_overlap": overlap_all,
                "bm25_unique": len(bm25_docs - vector_docs),
                "vector_unique": len(vector_docs - bm25_docs)
            },
            "score_ranges": {
                "bm25": {
                    "min": min([r.score for r in bm25_results]) if bm25_results else 0,
                    "max": max([r.score for r in bm25_results]) if bm25_results else 0,
                    "avg": sum([r.score for r in bm25_results]) / len(bm25_results) if bm25_results else 0
                },
                "vector": {
                    "min": min([r.score for r in vector_results]) if vector_results else 0,
                    "max": max([r.score for r in vector_results]) if vector_results else 0,
                    "avg": sum([r.score for r in vector_results]) / len(vector_results) if vector_results else 0
                },
                "hybrid": {
                    "min": min([r.score for r in hybrid_results]) if hybrid_results else 0,
                    "max": max([r.score for r in hybrid_results]) if hybrid_results else 0,
                    "avg": sum([r.score for r in hybrid_results]) / len(hybrid_results) if hybrid_results else 0
                }
            }
        }
    
    def update_weights(self, bm25_weight: float, vector_weight: float) -> None:
        """Update the weights for BM25 and vector search fusion."""
        if bm25_weight < 0 or vector_weight < 0:
            raise ValueError("Weights must be non-negative")
        
        total_weight = bm25_weight + vector_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be positive")
        
        # Normalize weights
        self.bm25_weight = bm25_weight / total_weight
        self.vector_weight = vector_weight / total_weight
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the hybrid search engine."""
        status = {
            "initialized": self.is_initialized,
            "bm25_ready": self.bm25_retriever.is_fitted,
            "vector_store_ready": self.vector_store_manager.vector_store is not None,
            "document_count": len(self.bm25_retriever.documents) if self.bm25_retriever.is_fitted else 0,
            "vocabulary_size": len(self.bm25_retriever.vocabulary) if self.bm25_retriever.is_fitted else 0,
            "weights": {
                "bm25": self.bm25_weight,
                "vector": self.vector_weight
            },
            "query_expansion_enabled": self.enable_query_expansion,
            "rrf_parameter": self.rrf.k,
            "reranking_enabled": self.enable_reranking
        }
        
        # Add reranker status if available
        if self.reranker:
            try:
                status["reranker_status"] = self.reranker.get_statistics()
            except:
                status["reranker_status"] = {"available": True, "error": "Failed to get statistics"}
        else:
            status["reranker_status"] = {"available": False}
        
        return status