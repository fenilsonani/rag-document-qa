"""
Enhanced RAG System with Offline Capabilities
Combines API-based and offline functionality for maximum flexibility.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from langchain.schema import Document

from .config import Config
from .rag_chain import RAGChain
from .offline_rag import OfflineRAG
from .document_intelligence import DocumentIntelligence
from .cross_reference_engine import CrossReferenceEngine
from .query_intelligence import QueryIntelligence
from .advanced_query_intelligence import AdvancedQueryIntelligence
from .hybrid_search import HybridSearchEngine
from .vector_store import VectorStoreManager
from .evaluation import RAGEvaluator
from .adaptive_chunking import AdaptiveChunker
from .redis_cache import SmartCache, cached_method
from .multimodal_rag import MultiModalRAG
from .graph_enhanced_rag import GraphEnhancedRAG


class EnhancedRAG:
    """Enhanced RAG system with both online and offline capabilities."""
    
    def __init__(self):
        self.config = Config()
        self.online_rag = RAGChain()
        self.offline_rag = OfflineRAG()
        self.model_manager = self.offline_rag.model_manager
        self.doc_intelligence = DocumentIntelligence()
        self.cross_ref_engine = CrossReferenceEngine()
        self.query_intelligence = QueryIntelligence()
        self.advanced_query_intelligence = AdvancedQueryIntelligence()
        
        # Initialize hybrid search components
        self.vector_store_manager = VectorStoreManager()
        self.hybrid_search = HybridSearchEngine(self.vector_store_manager)
        
        # Initialize evaluation framework
        self.evaluator = RAGEvaluator()
        
        # Initialize adaptive chunking system
        self.adaptive_chunker = AdaptiveChunker(self.config)
        
        # Initialize caching system
        self._cache = SmartCache(
            self.config,
            redis_host=self.config.REDIS_HOST,
            redis_port=self.config.REDIS_PORT,
            redis_db=self.config.REDIS_DB,
            redis_password=self.config.REDIS_PASSWORD
        )
        
        # Initialize multi-modal RAG system
        self.multimodal_rag = MultiModalRAG(self.config)
        
        # Initialize graph-enhanced RAG system
        self.graph_rag = GraphEnhancedRAG(self.config)
        
        self.mode = "offline"  # Default to offline mode
        self.initialized = False
        self.processed_documents = []
        self.adaptive_chunks = []
        self.hybrid_search_enabled = True
        self.evaluation_enabled = True
        self.adaptive_chunking_enabled = True
        self.caching_enabled = True
        self.multimodal_enabled = True
        self.graph_rag_enabled = True
        
    def initialize(self, force_mode: str = None) -> Dict[str, Any]:
        """Initialize the enhanced RAG system.
        
        Args:
            force_mode: Force specific mode ('online', 'offline', or None for auto)
        """
        results = {"online": None, "offline": None, "intelligence": True}
        
        # Check what modes are available
        online_available = self.config.validate_api_keys()
        
        # Determine target mode
        if force_mode == "online" and not online_available:
            return {
                "success": False,
                "error": "Online mode requested but no API keys available",
                "mode": "error"
            }
        
        target_mode = force_mode if force_mode else ("online" if online_available else "offline")
        
        # Initialize based on target mode
        if target_mode == "online" and online_available:
            try:
                online_result = self.online_rag.initialize()
                if online_result.get("success"):
                    self.mode = "online"
                    results["online"] = online_result
                    print("✅ Online mode with API keys enabled")
                else:
                    results["online"] = online_result
                    # Fallback to offline if online fails and not forced
                    if force_mode != "online":
                        target_mode = "offline"
            except Exception as e:
                results["online"] = {"success": False, "error": str(e)}
                if force_mode != "online":
                    target_mode = "offline"
        
        # Initialize offline mode if needed
        if target_mode == "offline" or self.mode != "online":
            try:
                offline_result = self.offline_rag.initialize()
                results["offline"] = offline_result
                if offline_result.get("success"):
                    self.mode = "offline"
                    print("🔧 Running in offline mode (no API keys needed)")
                else:
                    self.mode = "basic"
                    print("⚠️ Running in basic mode (limited functionality)")
            except Exception as e:
                results["offline"] = {"success": False, "error": str(e)}
                self.mode = "basic"
                print("⚠️ Running in basic mode (limited functionality)")
        
        self.initialized = True
        
        return {
            "success": True,
            "mode": self.mode,
            "online_available": online_available,
            "offline_available": True,  # Offline is always available
            "intelligence_available": True,
            "details": results
        }
    
    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process documents with intelligence analysis."""
        # Process documents based on mode
        if self.mode == "online":
            result = self.online_rag.process_documents(file_paths)
        else:
            # For offline mode, we need to load documents first
            from .document_loader import DocumentProcessor
            doc_processor = DocumentProcessor()
            
            try:
                documents = doc_processor.process_documents(file_paths)
                self.processed_documents = documents
                
                # Apply adaptive chunking if enabled
                if self.adaptive_chunking_enabled:
                    try:
                        self.adaptive_chunks = self.adaptive_chunker.chunk_documents(documents)
                        
                        # Convert adaptive chunks back to standard documents for offline RAG
                        chunked_documents = [chunk.to_langchain_document() for chunk in self.adaptive_chunks]
                        offline_result = self.offline_rag.process_documents(chunked_documents)
                        
                        result = {
                            "success": offline_result["success"],
                            "processed_files": len(file_paths),
                            "total_chunks": len(chunked_documents),
                            "adaptive_chunks": len(self.adaptive_chunks),
                            "processing_time": 1.0,  # Estimated
                            "mode": "offline",
                            "adaptive_chunking_applied": True
                        }
                        
                    except Exception as e:
                        print(f"⚠️ Adaptive chunking failed, falling back to standard chunking: {e}")
                        # Fallback to standard processing
                        offline_result = self.offline_rag.process_documents(documents)
                        result = {
                            "success": offline_result["success"],
                            "processed_files": len(file_paths),
                            "total_chunks": len(documents),
                            "processing_time": 1.0,
                            "mode": "offline",
                            "adaptive_chunking_applied": False
                        }
                else:
                    # Standard processing
                    offline_result = self.offline_rag.process_documents(documents)
                    result = {
                        "success": offline_result["success"],
                        "processed_files": len(file_paths),
                        "total_chunks": len(documents),
                        "processing_time": 1.0,
                        "mode": "offline",
                        "adaptive_chunking_applied": False
                    }
                
                if not offline_result["success"]:
                    result["error"] = offline_result.get("error", "Processing failed")
                    
            except Exception as e:
                result = {
                    "success": False,
                    "error": f"Document processing failed: {str(e)}"
                }
        
        # Add intelligence analysis if successful
        if result.get("success") and self.processed_documents:
            try:
                # Generate document insights
                insights = self.doc_intelligence.generate_document_insights(self.processed_documents)
                result["insights_generated"] = True
                result["insights"] = insights
                
                # Generate cross-reference analysis for multiple documents
                if len(self.processed_documents) > 1:
                    relationships = self.cross_ref_engine.analyze_document_relationships(self.processed_documents)
                    result["relationships_generated"] = True
                    result["relationships"] = relationships
                
                # Initialize hybrid search if enabled
                if self.hybrid_search_enabled:
                    try:
                        hybrid_result = self.hybrid_search.initialize(self.processed_documents)
                        result["hybrid_search_initialized"] = hybrid_result["success"]
                        if hybrid_result["success"]:
                            result["hybrid_search_info"] = {
                                "vocabulary_size": hybrid_result.get("bm25_vocabulary_size", 0),
                                "vector_store_ready": hybrid_result.get("vector_store_ready", False)
                            }
                    except Exception as e:
                        result["hybrid_search_error"] = str(e)
                        print(f"⚠️ Hybrid search initialization failed: {e}")
                
                # Process multi-modal elements if enabled
                if self.multimodal_enabled:
                    try:
                        multimodal_elements = []
                        for doc in self.processed_documents:
                            elements = self.multimodal_rag.process_document(doc)
                            multimodal_elements.extend(elements)
                        
                        result["multimodal_processing"] = {
                            "enabled": True,
                            "elements_extracted": len(multimodal_elements),
                            "element_types": {},
                            "capabilities": self.multimodal_rag.get_capabilities()
                        }
                        
                        # Count element types
                        for element in multimodal_elements:
                            element_type = element.element_type
                            result["multimodal_processing"]["element_types"][element_type] = (
                                result["multimodal_processing"]["element_types"].get(element_type, 0) + 1
                            )
                        
                        if multimodal_elements:
                            print(f"✅ Extracted {len(multimodal_elements)} multi-modal elements")
                        
                    except Exception as e:
                        result["multimodal_error"] = str(e)
                        print(f"⚠️ Multi-modal processing failed: {e}")
                
                # Build knowledge graph if enabled
                if self.graph_rag_enabled:
                    try:
                        graph_stats = self.graph_rag.process_documents(self.processed_documents)
                        
                        result["knowledge_graph"] = {
                            "enabled": True,
                            "documents_processed": graph_stats["documents_processed"],
                            "entities_extracted": graph_stats["entities_extracted"],
                            "relations_extracted": graph_stats["relations_extracted"],
                            "processing_time": graph_stats["processing_time"],
                            "graph_statistics": self.graph_rag.knowledge_graph.get_graph_statistics(),
                            "capabilities": self.graph_rag.get_capabilities()
                        }
                        
                        if graph_stats["entities_extracted"] > 0:
                            print(f"✅ Built knowledge graph with {graph_stats['entities_extracted']} entities and {graph_stats['relations_extracted']} relations")
                        
                    except Exception as e:
                        result["graph_rag_error"] = str(e)
                        print(f"⚠️ Knowledge graph construction failed: {e}")
                
            except Exception as e:
                result["intelligence_error"] = str(e)
                print(f"⚠️ Intelligence analysis failed: {e}")
        
        return result
    
    @cached_method('query_results', ttl=3600)
    def ask_question(self, question: str, use_conversation: bool = False) -> Dict[str, Any]:
        """Ask question with intelligent enhancement."""
        # Enhanced query analysis and rewriting
        enhanced_query = None
        advanced_analysis = None
        
        if self.processed_documents:
            try:
                # Basic query enhancement
                enhanced_query = self.query_intelligence.enhance_query(
                    question, self.processed_documents, []
                )
                
                # Advanced query intelligence processing
                context = {
                    'document_themes': self._extract_document_context(),
                    'main_concept': self._extract_main_concept()
                }
                
                advanced_analysis = self.advanced_query_intelligence.process_query(
                    question, context
                )
                
            except Exception as e:
                print(f"⚠️ Query enhancement failed: {e}")
        
        # Ask question based on mode
        if self.mode == "online":
            result = self.online_rag.ask_question(question, use_conversation)
        elif self.mode == "offline":
            result = self.offline_rag.ask_question(question)
        else:
            # Basic fallback
            result = {
                "success": False,
                "error": "No functional RAG mode available"
            }
        
        # Add query enhancement info
        if enhanced_query:
            result["query_enhancement"] = enhanced_query
        
        # Add advanced query intelligence results
        if advanced_analysis:
            result["advanced_query_analysis"] = advanced_analysis
            
            # Use recommended query if available and confidence is high
            recommended_query = advanced_analysis.get("recommended_query")
            if recommended_query and recommended_query != question and advanced_analysis.get("confidence_score", 0) > 0.7:
                print(f"🔍 Using enhanced query: {recommended_query}")
                # Re-run with enhanced query if it's significantly different
                if self.mode == "online":
                    enhanced_result = self.online_rag.ask_question(recommended_query, use_conversation)
                elif self.mode == "offline":
                    enhanced_result = self.offline_rag.ask_question(recommended_query)
                else:
                    enhanced_result = result
                
                if enhanced_result.get("success"):
                    result["enhanced_answer"] = enhanced_result.get("answer")
                    result["query_rewrite_applied"] = True
        
        return result
    
    def search_documents(self, query: str, k: int = 4, method: str = 'hybrid') -> Dict[str, Any]:
        """Search documents with multiple methods available."""
        try:
            # Use hybrid search if available and requested
            if self.hybrid_search_enabled and self.hybrid_search.is_initialized and method in ['hybrid', 'bm25', 'vector']:
                return self.search_documents_hybrid(query, k, method)
            
            # Fallback to mode-specific search
            if self.mode == "online":
                return self.online_rag.search_documents(query, k)
            elif self.mode == "offline":
                return self.offline_rag.search_documents(query, k)
            else:
                return {"success": False, "error": "No search capability available"}
                
        except Exception as e:
            return {"success": False, "error": f"Search failed: {str(e)}"}
    
    @cached_method('search_results', ttl=1800)
    def search_documents_hybrid(
        self, 
        query: str, 
        k: int = 4, 
        method: str = 'hybrid',
        enable_reranking: bool = True
    ) -> Dict[str, Any]:
        """Search documents using advanced hybrid search with optional reranking."""
        if not self.hybrid_search.is_initialized:
            return {"success": False, "error": "Hybrid search not initialized"}
        
        try:
            # Perform hybrid search with reranking
            search_results = self.hybrid_search.search(
                query=query, 
                k=k, 
                method=method,
                enable_reranking=enable_reranking
            )
            
            # Convert to expected format
            results = []
            for result in search_results:
                result_data = {
                    "content": result.document.page_content,
                    "metadata": result.document.metadata,
                    "score": result.score,
                    "rank": result.rank,
                    "source": result.source,
                    "search_metadata": result.metadata
                }
                
                # Add reranking information if available
                if 'rerank_confidence' in result.metadata:
                    result_data["reranking_info"] = {
                        "original_score": result.metadata.get('original_score'),
                        "original_rank": result.metadata.get('original_rank'),
                        "confidence": result.metadata.get('rerank_confidence'),
                        "score_improvement": result.metadata.get('score_improvement')
                    }
                
                results.append(result_data)
            
            return {
                "success": True,
                "query": query,
                "method": method,
                "results": results,
                "result_count": len(results),
                "reranking_applied": enable_reranking and any('rerank_confidence' in r["search_metadata"] for r in results)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Hybrid search failed: {str(e)}"}
    
    def get_search_analytics(self, query: str, k: int = 4) -> Dict[str, Any]:
        """Get detailed search analytics comparing different methods."""
        if not self.hybrid_search.is_initialized:
            return {"success": False, "error": "Hybrid search not initialized"}
        
        try:
            analytics = self.hybrid_search.get_search_analytics(query, k)
            return {
                "success": True,
                "analytics": analytics
            }
        except Exception as e:
            return {"success": False, "error": f"Analytics failed: {str(e)}"}
    
    @cached_method('reranking_scores', ttl=1800)
    def rerank_results(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply reranking to existing search results."""
        if not self.hybrid_search.is_initialized or not self.hybrid_search.reranker:
            return {"success": False, "error": "Reranker not available"}
        
        try:
            from .hybrid_search import SearchResult
            from langchain.schema import Document
            
            # Convert search results to SearchResult format
            search_result_objects = []
            for i, result in enumerate(search_results):
                doc = Document(
                    page_content=result["content"],
                    metadata=result.get("metadata", {})
                )
                search_result = SearchResult(
                    document=doc,
                    score=result.get("score", 0.0),
                    rank=result.get("rank", i + 1),
                    source=result.get("source", "unknown"),
                    metadata=result.get("search_metadata", {})
                )
                search_result_objects.append(search_result)
            
            # Apply reranking
            reranked = self.hybrid_search.reranker.rerank_with_explanation(
                query=query,
                search_results=search_result_objects
            )
            
            # Convert back to expected format
            reranked_results = []
            for rerank_result in reranked["reranked_results"]:
                reranked_results.append({
                    "content": rerank_result.document.page_content,
                    "metadata": rerank_result.document.metadata,
                    "score": rerank_result.rerank_score,
                    "rank": rerank_result.new_rank,
                    "source": rerank_result.source,
                    "reranking_info": {
                        "original_score": rerank_result.original_score,
                        "original_rank": rerank_result.original_rank,
                        "confidence": rerank_result.confidence,
                        "score_improvement": rerank_result.score_improvement
                    }
                })
            
            return {
                "success": True,
                "reranked_results": reranked_results,
                "analytics": reranked["analytics"],
                "query": query
            }
            
        except Exception as e:
            return {"success": False, "error": f"Reranking failed: {str(e)}"}
    
    def evaluate_query(
        self,
        query: str,
        ground_truth_answer: Optional[str] = None,
        relevant_document_ids: Optional[List[str]] = None,
        method: str = 'hybrid',
        enable_reranking: bool = True
    ) -> Dict[str, Any]:
        """Evaluate a single query using the comprehensive evaluation framework."""
        if not self.evaluation_enabled:
            return {"success": False, "error": "Evaluation framework not enabled"}
        
        try:
            import time
            start_time = time.time()
            
            # Get RAG response
            response = self.ask_question(query)
            
            if not response.get("success"):
                return {"success": False, "error": "Query failed", "details": response}
            
            # Get search results for evaluation
            search_results = self.search_documents(query, k=10, method=method)
            
            if not search_results.get("success"):
                return {"success": False, "error": "Search failed", "details": search_results}
            
            # Prepare data for evaluation
            retrieved_documents = []
            for i, result in enumerate(search_results["results"]):
                retrieved_documents.append({
                    "id": str(i),
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "score": result["score"]
                })
            
            # Use provided relevant docs or estimate from high scores
            if relevant_document_ids is None:
                # Simple heuristic: top 30% of results or score > 0.5
                threshold = max(0.5, np.percentile([r["score"] for r in search_results["results"]], 70))
                relevant_document_ids = [
                    str(i) for i, result in enumerate(search_results["results"])
                    if result["score"] > threshold
                ]
            
            total_time = time.time() - start_time
            
            # Perform evaluation
            evaluation_result = self.evaluator.evaluate_single_query(
                query=query,
                generated_answer=response.get("answer", ""),
                retrieved_documents=retrieved_documents,
                relevant_document_ids=relevant_document_ids,
                ground_truth_answer=ground_truth_answer,
                retrieval_time=search_results.get("processing_time", 0.0),
                generation_time=response.get("response_time", 0.0),
                metadata={
                    "method": method,
                    "reranking_enabled": enable_reranking,
                    "mode": self.mode
                }
            )
            
            return {
                "success": True,
                "evaluation_result": evaluation_result,
                "query": query,
                "generated_answer": response.get("answer", ""),
                "evaluation_time": total_time
            }
            
        except Exception as e:
            return {"success": False, "error": f"Evaluation failed: {str(e)}"}
    
    def run_evaluation_suite(
        self,
        evaluation_dataset: List[Dict[str, Any]],
        method: str = 'hybrid',
        enable_reranking: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on a dataset."""
        if not self.evaluation_enabled:
            return {"success": False, "error": "Evaluation framework not enabled"}
        
        try:
            def rag_callable(query: str) -> Dict[str, Any]:
                """Callable for the evaluator."""
                response = self.ask_question(query)
                search_results = self.search_documents(query, k=10, method=method)
                
                sources = []
                if search_results.get("success"):
                    for i, result in enumerate(search_results["results"]):
                        sources.append({
                            "id": str(i),
                            "content": result["content"],
                            "metadata": result["metadata"]
                        })
                
                return {
                    "answer": response.get("answer", ""),
                    "sources": sources,
                    "response_time": response.get("response_time", 0.0),
                    "retrieval_time": search_results.get("processing_time", 0.0)
                }
            
            # Run evaluation
            results = self.evaluator.evaluate_dataset(
                evaluation_dataset=evaluation_dataset,
                rag_system_callable=rag_callable,
                progress_callback=lambda current, total: print(f"Progress: {current}/{total}")
            )
            
            return {
                "success": True,
                "evaluation_results": results,
                "method": method,
                "reranking_enabled": enable_reranking
            }
            
        except Exception as e:
            return {"success": False, "error": f"Evaluation suite failed: {str(e)}"}
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary and performance metrics."""
        if not self.evaluation_enabled:
            return {"success": False, "error": "Evaluation framework not enabled"}
        
        try:
            summary = self.evaluator.get_summary_report()
            return {
                "success": True,
                "summary": summary
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get evaluation summary: {str(e)}"}
    
    def export_evaluation_results(self, filepath: str) -> Dict[str, Any]:
        """Export evaluation results to file."""
        if not self.evaluation_enabled:
            return {"success": False, "error": "Evaluation framework not enabled"}
        
        try:
            self.evaluator.export_results(filepath)
            return {"success": True, "filepath": filepath}
        except Exception as e:
            return {"success": False, "error": f"Export failed: {str(e)}"}
    
    @cached_method('document_analysis', ttl=14400)
    def get_document_insights(self) -> Optional[Dict[str, Any]]:
        """Get document intelligence insights."""
        if self.processed_documents:
            try:
                return self.doc_intelligence.generate_document_insights(self.processed_documents)
            except Exception as e:
                return {"error": f"Intelligence analysis failed: {str(e)}"}
        return None
    
    @cached_method('document_analysis', ttl=14400)
    def get_relationship_analysis(self) -> Optional[Dict[str, Any]]:
        """Get cross-reference relationship analysis."""
        if len(self.processed_documents) > 1:
            try:
                return self.cross_ref_engine.analyze_document_relationships(self.processed_documents)
            except Exception as e:
                return {"error": f"Relationship analysis failed: {str(e)}"}
        return None
    
    def get_smart_suggestions(self) -> List[Dict[str, Any]]:
        """Get smart question suggestions."""
        if not self.processed_documents:
            return []
        
        try:
            # Generate suggestions based on document content
            suggestions = []
            
            # Get key concepts for suggestions
            insights = self.doc_intelligence.generate_document_insights(self.processed_documents)
            concepts = insights.get('key_concepts', [])
            
            # Generate different types of questions
            if concepts:
                # Definition questions
                for concept in concepts[:3]:
                    suggestions.append({
                        "question": f"What is {concept['concept']}?",
                        "type": "definition",
                        "confidence": 0.9
                    })
                
                # Relationship questions
                if len(concepts) > 1:
                    suggestions.append({
                        "question": f"How does {concepts[0]['concept']} relate to {concepts[1]['concept']}?",
                        "type": "relationship",
                        "confidence": 0.8
                    })
                
                # Analysis questions
                suggestions.append({
                    "question": f"What are the main points about {concepts[0]['concept']}?",
                    "type": "analysis",
                    "confidence": 0.8
                })
            
            # Add document-specific suggestions
            doc_types = set()
            for doc in self.processed_documents:
                file_type = doc.metadata.get('file_type', '')
                if file_type:
                    doc_types.add(file_type)
            
            if len(self.processed_documents) > 1:
                suggestions.append({
                    "question": "What are the main differences between these documents?",
                    "type": "comparison",
                    "confidence": 0.7
                })
                
                suggestions.append({
                    "question": "Summarize the key points from all documents",
                    "type": "summary",
                    "confidence": 0.8
                })
            
            return suggestions[:6]  # Return top 6 suggestions
            
        except Exception as e:
            print(f"⚠️ Smart suggestions failed: {e}")
            return [
                {"question": "What are the main topics in these documents?", "type": "general", "confidence": 0.5},
                {"question": "Can you summarize the key points?", "type": "summary", "confidence": 0.5},
                {"question": "What are the most important concepts discussed?", "type": "analysis", "confidence": 0.5}
            ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "mode": self.mode,
            "initialized": self.initialized,
            "documents_processed": len(self.processed_documents),
            "api_keys_available": self.config.validate_api_keys(),
            "capabilities": {
                "question_answering": self.mode in ["online", "offline"],
                "document_intelligence": True,
                "cross_referencing": len(self.processed_documents) > 1,
                "smart_suggestions": len(self.processed_documents) > 0,
                "hybrid_search": self.hybrid_search_enabled and self.hybrid_search.is_initialized,
                "advanced_search_methods": ["hybrid", "bm25", "vector"] if self.hybrid_search_enabled and self.hybrid_search.is_initialized else ["basic"],
                "comprehensive_evaluation": self.evaluation_enabled,
                "reranking": self.hybrid_search_enabled and self.hybrid_search.reranker is not None,
                "adaptive_chunking": self.adaptive_chunking_enabled,
                "structure_aware_processing": len(self.adaptive_chunks) > 0,
                "high_performance_caching": self.caching_enabled,
                "redis_cache": self._cache.is_redis_available() if self.caching_enabled else False,
                "multimodal_processing": self.multimodal_enabled,
                "table_processing": self.multimodal_rag.get_capabilities().get("table_processing", False) if self.multimodal_enabled else False,
                "image_processing": self.multimodal_rag.get_capabilities().get("image_processing", False) if self.multimodal_enabled else False,
                "ai_image_understanding": self.multimodal_rag.get_capabilities().get("ai_image_understanding", False) if self.multimodal_enabled else False,
                "graph_enhanced_rag": self.graph_rag_enabled,
                "entity_extraction": self.graph_rag.get_capabilities().get("entity_extraction", False) if self.graph_rag_enabled else False,
                "relation_extraction": self.graph_rag.get_capabilities().get("relation_extraction", False) if self.graph_rag_enabled else False,
                "named_entity_recognition": self.graph_rag.get_capabilities().get("named_entity_recognition", False) if self.graph_rag_enabled else False
            }
        }
        
        # Add hybrid search status
        if self.hybrid_search_enabled:
            status["hybrid_search_status"] = self.hybrid_search.get_status()
        
        # Add evaluation status
        if self.evaluation_enabled:
            status["evaluation_status"] = {
                "enabled": True,
                "total_evaluations": len(self.evaluator.evaluation_history),
                "recent_evaluations": len([
                    r for r in self.evaluator.evaluation_history[-10:]
                ]) if self.evaluator.evaluation_history else 0
            }
        
        # Add adaptive chunking status
        if self.adaptive_chunking_enabled:
            status["adaptive_chunking_status"] = {
                "enabled": True,
                "total_adaptive_chunks": len(self.adaptive_chunks),
                "chunk_types": len(set(chunk.metadata.chunk_type for chunk in self.adaptive_chunks)) if self.adaptive_chunks else 0,
                "avg_chunk_confidence": np.mean([chunk.confidence_score for chunk in self.adaptive_chunks]) if self.adaptive_chunks else 0,
                "structure_aware": len([chunk for chunk in self.adaptive_chunks if chunk.metadata.structure_level > 0]) if self.adaptive_chunks else 0
            }
        
        # Add cache status
        if self.caching_enabled:
            cache_stats = self.get_cache_stats()
            if cache_stats.get("success"):
                cache_data = cache_stats["stats"]
                metrics = cache_data.get("metrics", {})
                
                status["cache_status"] = {
                    "enabled": True,
                    "cache_type": cache_data.get("cache_type", "unknown"),
                    "redis_available": cache_data.get("redis_available", False),
                    "total_requests": metrics.get("total_requests", 0),
                    "cache_hits": metrics.get("cache_hits", 0),
                    "cache_misses": metrics.get("cache_misses", 0),
                    "hit_rate": metrics.get("hit_rate", 0.0),
                    "avg_retrieval_time": metrics.get("avg_retrieval_time", 0.0),
                    "total_time_saved": metrics.get("total_time_saved", 0.0),
                    "namespaces": len(cache_data.get("namespaces", {}))
                }
            else:
                status["cache_status"] = {"enabled": False, "error": cache_stats.get("error", "Unknown")}
        
        # Add multi-modal status
        if self.multimodal_enabled:
            multimodal_summary = self.get_multimodal_summary()
            if multimodal_summary.get("success"):
                summary_data = multimodal_summary["summary"]
                capabilities = multimodal_summary["capabilities"]
                
                status["multimodal_status"] = {
                    "enabled": True,
                    "total_elements": summary_data.get("total_elements", 0),
                    "element_types": summary_data.get("element_types", {}),
                    "capabilities": capabilities,
                    "processing_methods": summary_data.get("processing_methods", {}),
                    "average_confidence": summary_data.get("average_confidence", 0.0)
                }
            else:
                status["multimodal_status"] = {"enabled": False, "error": multimodal_summary.get("error", "Unknown")}
        
        # Add knowledge graph status
        if self.graph_rag_enabled:
            try:
                graph_stats = self.graph_rag.knowledge_graph.get_graph_statistics()
                processing_stats = self.graph_rag.processing_stats
                
                status["knowledge_graph"] = {
                    "enabled": True,
                    "documents_processed": processing_stats.get("documents_processed", 0),
                    "entities_extracted": processing_stats.get("entities_extracted", 0),
                    "relations_extracted": processing_stats.get("relations_extracted", 0),
                    "processing_time": processing_stats.get("processing_time", 0.0),
                    "graph_statistics": graph_stats,
                    "capabilities": self.graph_rag.get_capabilities()
                }
            except Exception as e:
                status["knowledge_graph"] = {"enabled": False, "error": str(e)}
        
        if self.mode == "online":
            online_status = self.online_rag.get_system_status()
            status.update(online_status)
        elif self.mode == "offline":
            offline_status = self.offline_rag.get_status()
            status["offline_status"] = offline_status
        
        return status
    
    def get_model_manager_status(self) -> Dict[str, Any]:
        """Get model manager status for offline mode."""
        if self.mode != "offline":
            return {"success": False, "error": "Model manager only available in offline mode"}
        
        try:
            return {
                "success": True,
                "model_status": self.offline_rag.get_model_status(),
                "storage_info": self.model_manager.get_storage_info(),
                "available_models": {
                    task: {key: info.name for key, info in models.items()}
                    for task, models in {
                        "question-answering": self.model_manager.get_models_by_task("question-answering"),
                        "summarization": self.model_manager.get_models_by_task("summarization"),
                        "embedding": self.model_manager.get_models_by_task("embedding")
                    }.items()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_document_context(self) -> Dict[str, List[str]]:
        """Extract document themes for advanced query intelligence context."""
        if not self.processed_documents:
            return {}
        
        try:
            insights = self.doc_intelligence.generate_document_insights(self.processed_documents)
            themes = {}
            
            # Extract key concepts as themes
            if 'key_concepts' in insights:
                concepts = [concept['concept'] for concept in insights['key_concepts'][:10]]
                themes['main_concepts'] = concepts
            
            # Extract document types
            doc_types = set()
            for doc in self.processed_documents:
                file_type = doc.metadata.get('file_type', '')
                if file_type:
                    doc_types.add(file_type)
            
            if doc_types:
                themes['document_types'] = list(doc_types)
            
            return themes
            
        except Exception as e:
            print(f"⚠️ Document context extraction failed: {e}")
            return {}
    
    def _extract_main_concept(self) -> Optional[str]:
        """Extract the main concept from processed documents."""
        if not self.processed_documents:
            return None
        
        try:
            insights = self.doc_intelligence.generate_document_insights(self.processed_documents)
            if 'key_concepts' in insights and insights['key_concepts']:
                return insights['key_concepts'][0]['concept']
        except Exception:
            pass
        
        return None
    
    def get_query_analytics(self) -> Dict[str, Any]:
        """Get analytics about query processing patterns."""
        try:
            analytics = self.advanced_query_intelligence.get_query_analytics()
            return {"success": True, "analytics": analytics}
        except Exception as e:
            return {"success": False, "error": f"Analytics failed: {str(e)}"}
    
    def get_adaptive_chunking_analytics(self) -> Dict[str, Any]:
        """Get analytics about the adaptive chunking process."""
        if not self.adaptive_chunking_enabled:
            return {"success": False, "error": "Adaptive chunking not enabled"}
        
        if not self.adaptive_chunks:
            return {"success": False, "error": "No adaptive chunks available"}
        
        try:
            analytics = self.adaptive_chunker.get_chunking_analytics(self.adaptive_chunks)
            return {"success": True, "analytics": analytics}
        except Exception as e:
            return {"success": False, "error": f"Adaptive chunking analytics failed: {str(e)}"}
    
    def get_chunk_details(self, chunk_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about adaptive chunks."""
        if not self.adaptive_chunks:
            return {"success": False, "error": "No adaptive chunks available"}
        
        try:
            if chunk_id:
                # Find specific chunk
                target_chunk = None
                for chunk in self.adaptive_chunks:
                    if chunk.metadata.chunk_id == chunk_id:
                        target_chunk = chunk
                        break
                
                if not target_chunk:
                    return {"success": False, "error": f"Chunk {chunk_id} not found"}
                
                return {
                    "success": True,
                    "chunk": {
                        "id": target_chunk.metadata.chunk_id,
                        "type": target_chunk.metadata.chunk_type,
                        "content": target_chunk.content[:500] + "..." if len(target_chunk.content) > 500 else target_chunk.content,
                        "size": target_chunk.chunk_size,
                        "structure_level": target_chunk.metadata.structure_level,
                        "semantic_coherence": target_chunk.metadata.semantic_coherence,
                        "content_complexity": target_chunk.metadata.content_complexity,
                        "confidence_score": target_chunk.confidence_score,
                        "related_chunks": target_chunk.related_chunks,
                        "parent_section": target_chunk.metadata.parent_section
                    }
                }
            else:
                # Return summary of all chunks
                chunk_summaries = []
                for chunk in self.adaptive_chunks[:20]:  # Limit to first 20 for display
                    chunk_summaries.append({
                        "id": chunk.metadata.chunk_id,
                        "type": chunk.metadata.chunk_type,
                        "size": chunk.chunk_size,
                        "coherence": chunk.metadata.semantic_coherence,
                        "complexity": chunk.metadata.content_complexity,
                        "confidence": chunk.confidence_score,
                        "preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    })
                
                return {
                    "success": True,
                    "total_chunks": len(self.adaptive_chunks),
                    "displayed_chunks": len(chunk_summaries),
                    "chunks": chunk_summaries
                }
        
        except Exception as e:
            return {"success": False, "error": f"Failed to get chunk details: {str(e)}"}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.caching_enabled or not self._cache:
            return {"success": False, "error": "Caching not enabled"}
        
        try:
            stats = self._cache.get_stats()
            return {"success": True, "stats": stats}
        except Exception as e:
            return {"success": False, "error": f"Failed to get cache stats: {str(e)}"}
    
    def clear_cache(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Clear cache entries."""
        if not self.caching_enabled or not self._cache:
            return {"success": False, "error": "Caching not enabled"}
        
        try:
            if namespace:
                cleared = self._cache.clear_namespace(namespace)
                return {"success": True, "cleared_entries": cleared, "namespace": namespace}
            else:
                # Clear all RAG-related namespaces
                namespaces = ['query_results', 'search_results', 'document_analysis', 'reranking_scores', 'embeddings']
                total_cleared = 0
                for ns in namespaces:
                    total_cleared += self._cache.clear_namespace(ns)
                
                return {"success": True, "cleared_entries": total_cleared, "namespaces_cleared": namespaces}
        
        except Exception as e:
            return {"success": False, "error": f"Failed to clear cache: {str(e)}"}
    
    def warm_cache(self, common_queries: List[str]) -> Dict[str, Any]:
        """Pre-populate cache with common queries for better performance."""
        if not self.caching_enabled or not self._cache:
            return {"success": False, "error": "Caching not enabled"}
        
        if not self.processed_documents:
            return {"success": False, "error": "No documents processed yet"}
        
        try:
            warmed_count = 0
            failed_count = 0
            
            for query in common_queries:
                try:
                    # Perform search to cache results
                    search_result = self.search_documents(query, k=5)
                    if search_result.get("success"):
                        warmed_count += 1
                    
                    # Also try Q&A to cache responses
                    qa_result = self.ask_question(query)
                    if qa_result.get("success"):
                        warmed_count += 1
                
                except Exception as e:
                    failed_count += 1
                    logging.warning(f"Cache warming failed for query '{query}': {e}")
            
            return {
                "success": True,
                "queries_processed": len(common_queries),
                "cache_entries_created": warmed_count,
                "failed_queries": failed_count
            }
        
        except Exception as e:
            return {"success": False, "error": f"Cache warming failed: {str(e)}"}
    
    def query_multimodal_elements(self, query: str, element_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query multi-modal elements like tables, images, and charts."""
        if not self.multimodal_enabled:
            return {"success": False, "error": "Multi-modal processing not enabled"}
        
        try:
            elements = self.multimodal_rag.query_multimodal_elements(query, element_types)
            
            # Convert elements to serializable format
            results = []
            for element in elements:
                element_data = {
                    "element_id": element.element_id,
                    "element_type": element.element_type,
                    "text_description": element.text_description,
                    "confidence_score": element.confidence_score,
                    "processing_method": element.processing_method,
                    "metadata": element.metadata
                }
                
                # Add type-specific information
                if element.element_type == "table" and element.structured_data:
                    element_data["table_info"] = {
                        "num_rows": len(element.structured_data.get("data", [])),
                        "num_columns": len(element.structured_data.get("columns", [])),
                        "columns": list(element.structured_data.get("columns", []))
                    }
                
                elif element.element_type in ["image", "chart"]:
                    if "analysis" in element.metadata:
                        analysis = element.metadata["analysis"]
                        element_data["image_info"] = {
                            "dimensions": f"{analysis.get('width', 0)}x{analysis.get('height', 0)}",
                            "format": analysis.get("format", "Unknown"),
                            "detected_objects": len(analysis.get("objects_detected", [])),
                            "has_text": bool(analysis.get("detected_text", ""))
                        }
                
                results.append(element_data)
            
            return {
                "success": True,
                "query": query,
                "element_types_filter": element_types,
                "results": results,
                "total_found": len(results)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Multi-modal query failed: {str(e)}"}
    
    def get_multimodal_summary(self) -> Dict[str, Any]:
        """Get summary of all multi-modal elements."""
        if not self.multimodal_enabled:
            return {"success": False, "error": "Multi-modal processing not enabled"}
        
        try:
            summary = self.multimodal_rag.get_element_summary()
            capabilities = self.multimodal_rag.get_capabilities()
            
            return {
                "success": True,
                "summary": summary,
                "capabilities": capabilities,
                "processing_status": {
                    "enabled": True,
                    "total_elements": summary.get("total_elements", 0),
                    "last_processed": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Multi-modal summary failed: {str(e)}"}
    
    def export_multimodal_elements(self, filepath: str) -> Dict[str, Any]:
        """Export multi-modal elements to file."""
        if not self.multimodal_enabled:
            return {"success": False, "error": "Multi-modal processing not enabled"}
        
        try:
            success = self.multimodal_rag.export_elements(filepath)
            
            if success:
                return {"success": True, "filepath": filepath, "exported": True}
            else:
                return {"success": False, "error": "Export failed"}
                
        except Exception as e:
            return {"success": False, "error": f"Export failed: {str(e)}"}
    
    def query_knowledge_graph(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Query the knowledge graph for entity relationships and semantic connections."""
        if not self.graph_rag_enabled:
            return {"success": False, "error": "Knowledge graph not enabled"}
        
        try:
            graph_results = self.graph_rag.enhanced_query(query, top_k)
            
            result = {
                "success": True,
                "query": query,
                "graph_results": graph_results,
                "capabilities": self.graph_rag.get_capabilities()
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Knowledge graph query failed: {str(e)}"}
    
    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get comprehensive context and relationships for a specific entity."""
        if not self.graph_rag_enabled:
            return {"success": False, "error": "Knowledge graph not enabled"}
        
        try:
            context = self.graph_rag.get_entity_context(entity_name)
            
            if "error" in context:
                return {"success": False, "error": context["error"]}
            
            return {
                "success": True,
                "entity_name": entity_name,
                "context": context
            }
            
        except Exception as e:
            return {"success": False, "error": f"Entity context retrieval failed: {str(e)}"}
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        if not self.graph_rag_enabled:
            return {"success": False, "error": "Knowledge graph not enabled"}
        
        try:
            stats = self.graph_rag.knowledge_graph.get_graph_statistics()
            
            return {
                "success": True,
                "statistics": stats,
                "capabilities": self.graph_rag.get_capabilities()
            }
            
        except Exception as e:
            return {"success": False, "error": f"Graph statistics failed: {str(e)}"}
    
    def export_knowledge_graph(self, filepath: str, format: str = 'json') -> Dict[str, Any]:
        """Export the knowledge graph to a file."""
        if not self.graph_rag_enabled:
            return {"success": False, "error": "Knowledge graph not enabled"}
        
        try:
            success = self.graph_rag.knowledge_graph.export_graph(filepath, format)
            
            if success:
                return {"success": True, "filepath": filepath, "format": format, "exported": True}
            else:
                return {"success": False, "error": "Export failed"}
                
        except Exception as e:
            return {"success": False, "error": f"Graph export failed: {str(e)}"}
    
    def enhanced_search_with_graph(self, query: str, k: int = 4, include_graph: bool = True) -> Dict[str, Any]:
        """Enhanced search combining traditional RAG with knowledge graph insights."""
        try:
            # Perform standard search
            search_result = self.search_documents(query, k)
            
            # Add graph-enhanced results if enabled
            if include_graph and self.graph_rag_enabled:
                try:
                    graph_result = self.query_knowledge_graph(query, k)
                    
                    if graph_result.get("success"):
                        search_result["graph_enhanced"] = True
                        search_result["graph_insights"] = graph_result["graph_results"]
                        
                        # Merge and rank results based on relevance
                        combined_results = self._merge_search_and_graph_results(
                            search_result.get("results", []),
                            graph_result["graph_results"]
                        )
                        
                        search_result["combined_results"] = combined_results
                        search_result["total_sources"] = len(combined_results)
                    
                except Exception as e:
                    search_result["graph_error"] = str(e)
                    print(f"⚠️ Graph enhancement failed: {e}")
            
            return search_result
            
        except Exception as e:
            return {"success": False, "error": f"Enhanced search failed: {str(e)}"}
    
    def _merge_search_and_graph_results(self, search_results: List[Dict], graph_results: Dict) -> List[Dict]:
        """Merge and rank search results with graph insights."""
        try:
            combined = []
            
            # Add search results with source type
            for result in search_results:
                result["source_type"] = "document_search"
                result["relevance_score"] = result.get("score", 0.5)
                combined.append(result)
            
            # Add graph results
            for match in graph_results.get("direct_matches", []):
                entity_info = match.get("entity", {})
                
                graph_result = {
                    "content": entity_info.get("text_description", ""),
                    "metadata": {
                        "source": "knowledge_graph",
                        "entity_type": entity_info.get("entity_type", ""),
                        "entity_name": entity_info.get("name", ""),
                        "confidence": entity_info.get("confidence", 0.0)
                    },
                    "source_type": "knowledge_graph",
                    "relevance_score": entity_info.get("confidence", 0.0),
                    "score": entity_info.get("confidence", 0.0)
                }
                
                combined.append(graph_result)
            
            # Sort by relevance score
            combined.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            return combined[:10]  # Return top 10 combined results
            
        except Exception as e:
            logging.warning(f"Result merging failed: {e}")
            return search_results  # Return original search results on failure