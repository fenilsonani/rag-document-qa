"""
Offline RAG System - No API Keys Required
Uses local models for completely offline document Q&A functionality.
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from langchain.schema import Document
from .config import Config
from .model_manager import ModelManager


class OfflineRAG:
    """Complete offline RAG system using local models."""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = ModelManager()
        self.qa_pipeline = None
        self.summarizer = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_vectors = None
        self.documents = []
        self.initialized = False
        self.selected_models = {
            "qa_model": "qa_distilbert",
            "summarizer_model": "summarizer_t5",
            "embedder_model": "embedder_sentence"
        }
        
    def initialize(self, selected_models: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Initialize offline models with selected model configurations."""
        if not TRANSFORMERS_AVAILABLE:
            return {
                "success": False,
                "error": "transformers library not available",
                "fallback": "Using basic text processing"
            }
        
        if selected_models:
            self.selected_models.update(selected_models)
        
        try:
            print("🔄 Loading offline models (this may take a moment)...")
            models_loaded = []
            
            # Load question-answering model
            qa_model_key = self.selected_models.get("qa_model", "qa_distilbert")
            qa_model_info = self.model_manager.get_model_info(qa_model_key)
            
            if qa_model_info and qa_model_info.is_downloaded:
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=qa_model_info.local_path,
                    tokenizer=qa_model_info.local_path
                )
                models_loaded.append(f"{qa_model_info.name} (local)")
            else:
                # Fallback to default model
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad"
                )
                models_loaded.append("DistilBERT QA (downloaded)")
            
            # Load summarization model
            summarizer_model_key = self.selected_models.get("summarizer_model", "summarizer_t5")
            summarizer_model_info = self.model_manager.get_model_info(summarizer_model_key)
            
            if summarizer_model_info and summarizer_model_info.is_downloaded:
                self.summarizer = pipeline(
                    "summarization",
                    model=summarizer_model_info.local_path,
                    tokenizer=summarizer_model_info.local_path,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                models_loaded.append(f"{summarizer_model_info.name} (local)")
            else:
                # Fallback to smaller model
                self.summarizer = pipeline(
                    "summarization",
                    model="t5-small",
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                models_loaded.append("T5-Small Summarizer (downloaded)")
            
            self.initialized = True
            print("✅ Offline models loaded successfully!")
            
            return {
                "success": True,
                "message": "Offline RAG system initialized",
                "models_loaded": models_loaded,
                "selected_models": self.selected_models
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize offline models: {str(e)}",
                "fallback": "Using basic text processing"
            }
    
    def set_selected_models(self, selected_models: Dict[str, str]):
        """Update selected models and reinitialize if needed."""
        self.selected_models.update(selected_models)
        if self.initialized:
            # Reinitialize with new models
            return self.initialize()
        return {"success": True, "message": "Models updated, call initialize() to load"}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of current models."""
        status = {
            "initialized": self.initialized,
            "selected_models": self.selected_models.copy(),
            "model_details": {},
            "storage_info": self.model_manager.get_storage_info()
        }
        
        for model_type, model_key in self.selected_models.items():
            model_info = self.model_manager.get_model_info(model_key)
            if model_info:
                status["model_details"][model_type] = {
                    "name": model_info.name,
                    "is_downloaded": model_info.is_downloaded,
                    "size_mb": model_info.size_mb,
                    "performance_score": model_info.performance_score,
                    "memory_usage": model_info.memory_usage
                }
        
        return status
    
    def process_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Process documents for offline RAG."""
        if not documents:
            return {"success": False, "error": "No documents provided"}
        
        try:
            self.documents = documents
            
            # Create document vectors
            texts = [doc.page_content for doc in documents]
            self.document_vectors = self.vectorizer.fit_transform(texts)
            
            return {
                "success": True,
                "processed_documents": len(documents),
                "vector_dimensions": self.document_vectors.shape[1] if self.document_vectors is not None else 0,
                "ready_for_qa": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Document processing failed: {str(e)}"
            }
    
    def ask_question(self, question: str, max_docs: int = 3) -> Dict[str, Any]:
        """Answer question using offline models."""
        if not self.documents or self.document_vectors is None:
            return {
                "success": False,
                "error": "No documents processed. Please process documents first."
            }
        
        try:
            start_time = datetime.now()
            
            # Find most relevant documents
            question_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, self.document_vectors)[0]
            
            # Get top documents
            top_indices = similarities.argsort()[-max_docs:][::-1]
            relevant_docs = [self.documents[i] for i in top_indices]
            relevance_scores = [similarities[i] for i in top_indices]
            
            # Combine relevant text
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Answer using offline QA model
            if self.qa_pipeline and len(context) > 0:
                # Truncate context if too long (BERT has token limits)
                max_context_length = 3000  # Approximate token limit
                if len(context) > max_context_length:
                    context = context[:max_context_length] + "..."
                
                try:
                    result = self.qa_pipeline(question=question, context=context)
                    answer = result['answer']
                    confidence = result['score']
                except Exception as e:
                    # Fallback to simple text matching
                    answer = self._fallback_answer(question, context)
                    confidence = 0.5
            else:
                # Fallback to simple text matching
                answer = self._fallback_answer(question, context)
                confidence = 0.3
            
            # Create source information
            sources = []
            for i, (doc, score) in enumerate(zip(relevant_docs, relevance_scores)):
                sources.append({
                    "index": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                    "filename": doc.metadata.get('filename', f'Document_{i}')
                })
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "response_time": round(response_time, 2),
                "method": "offline_models" if self.qa_pipeline else "text_matching"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Question answering failed: {str(e)}"
            }
    
    def _fallback_answer(self, question: str, context: str) -> str:
        """Fallback answer generation using simple text processing."""
        # Extract sentences that contain question keywords
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        sentences = re.split(r'[.!?]+', context)
        
        # Score sentences by keyword overlap
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
                
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words & sentence_words)
            
            if overlap > 0:
                scored_sentences.append((sentence.strip(), overlap))
        
        if scored_sentences:
            # Sort by overlap score and get top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, score in scored_sentences[:3]]
            return " ".join(top_sentences)
        else:
            # If no good match, return first meaningful paragraph
            paragraphs = context.split('\n\n')
            for para in paragraphs:
                if len(para.strip()) > 50:
                    return para.strip()[:300] + "..."
            
            return "I couldn't find a specific answer to your question in the provided documents."
    
    def generate_summary(self, text: str) -> str:
        """Generate summary using offline model or fallback."""
        try:
            if self.summarizer and len(text) > 100:
                # Truncate if too long
                max_input_length = 1000
                if len(text) > max_input_length:
                    text = text[:max_input_length]
                
                summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
                return summary[0]['summary_text']
            else:
                # Fallback: extract first few sentences
                sentences = re.split(r'[.!?]+', text)
                meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
                return ". ".join(meaningful_sentences[:3]) + "."
                
        except Exception as e:
            # Simple extractive summary
            sentences = re.split(r'[.!?]+', text)
            return ". ".join([s.strip() for s in sentences[:2] if len(s.strip()) > 20]) + "."
    
    def search_documents(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search documents without question answering."""
        if not self.documents or self.document_vectors is None:
            return {
                "success": False,
                "error": "No documents processed"
            }
        
        try:
            # Find similar documents
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get top k documents
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for i in top_indices:
                if similarities[i] > 0.01:  # Only include relevant results
                    results.append({
                        "content": self.documents[i].page_content,
                        "metadata": self.documents[i].metadata,
                        "similarity_score": float(similarities[i]),
                        "filename": self.documents[i].metadata.get('filename', f'Document_{i}')
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "result_count": len(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Search failed: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get offline system status."""
        return {
            "initialized": self.initialized,
            "models_loaded": {
                "qa_pipeline": self.qa_pipeline is not None,
                "summarizer": self.summarizer is not None
            },
            "documents_processed": len(self.documents),
            "vector_store_ready": self.document_vectors is not None,
            "mode": "offline"
        }