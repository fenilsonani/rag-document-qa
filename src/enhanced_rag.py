"""
Enhanced RAG System with Offline Capabilities
Combines API-based and offline functionality for maximum flexibility.
"""

import os
from typing import Dict, List, Any, Optional
from langchain.schema import Document

from .config import Config
from .rag_chain import RAGChain
from .offline_rag import OfflineRAG
from .document_intelligence import DocumentIntelligence
from .cross_reference_engine import CrossReferenceEngine
from .query_intelligence import QueryIntelligence


class EnhancedRAG:
    """Enhanced RAG system with both online and offline capabilities."""
    
    def __init__(self):
        self.config = Config()
        self.online_rag = RAGChain()
        self.offline_rag = OfflineRAG()
        self.doc_intelligence = DocumentIntelligence()
        self.cross_ref_engine = CrossReferenceEngine()
        self.query_intelligence = QueryIntelligence()
        
        self.mode = "offline"  # Default to offline mode
        self.initialized = False
        self.processed_documents = []
        
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
                
                # Process with offline RAG
                offline_result = self.offline_rag.process_documents(documents)
                
                result = {
                    "success": offline_result["success"],
                    "processed_files": len(file_paths),
                    "total_chunks": len(documents),
                    "processing_time": 1.0,  # Estimated
                    "mode": "offline"
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
                
            except Exception as e:
                result["intelligence_error"] = str(e)
                print(f"⚠️ Intelligence analysis failed: {e}")
        
        return result
    
    def ask_question(self, question: str, use_conversation: bool = False) -> Dict[str, Any]:
        """Ask question with intelligent enhancement."""
        # Enhance query first
        enhanced_query = None
        if self.processed_documents:
            try:
                enhanced_query = self.query_intelligence.enhance_query(
                    question, self.processed_documents, []
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
        
        return result
    
    def search_documents(self, query: str, k: int = 4) -> Dict[str, Any]:
        """Search documents."""
        if self.mode == "online":
            return self.online_rag.search_documents(query, k)
        elif self.mode == "offline":
            return self.offline_rag.search_documents(query, k)
        else:
            return {"success": False, "error": "No search capability available"}
    
    def get_document_insights(self) -> Optional[Dict[str, Any]]:
        """Get document intelligence insights."""
        if self.processed_documents:
            try:
                return self.doc_intelligence.generate_document_insights(self.processed_documents)
            except Exception as e:
                return {"error": f"Intelligence analysis failed: {str(e)}"}
        return None
    
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
                "smart_suggestions": len(self.processed_documents) > 0
            }
        }
        
        if self.mode == "online":
            online_status = self.online_rag.get_system_status()
            status.update(online_status)
        elif self.mode == "offline":
            offline_status = self.offline_rag.get_status()
            status["offline_status"] = offline_status
        
        return status