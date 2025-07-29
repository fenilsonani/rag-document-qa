"""
Comprehensive RAG Evaluation Framework - Pro-Level Assessment System
Implements sophisticated metrics for evaluating retrieval, generation, and end-to-end RAG performance.
"""

import json
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
import re
import logging

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Some evaluation metrics will be limited.")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Some text quality metrics will be limited.")

from langchain.schema import Document

from .config import Config


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    normalized_dcg_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    average_precision: float
    retrieval_latency: float
    coverage: float  # Proportion of relevant documents retrieved


@dataclass
class GenerationMetrics:
    """Metrics for evaluating generation quality."""
    faithfulness: float  # How well answer is grounded in sources
    answer_relevancy: float  # How relevant answer is to question
    context_precision: float  # Precision of retrieved context
    context_recall: float  # Recall of retrieved context
    context_relevancy: float  # Relevancy of retrieved context
    answer_correctness: float  # Overall answer correctness
    answer_semantic_similarity: float  # Semantic similarity to reference
    hallucination_score: float  # Likelihood of hallucination (lower is better)
    coherence_score: float  # Internal coherence of answer
    completeness_score: float  # How complete the answer is
    generation_latency: float


@dataclass
class QualityMetrics:
    """Text quality and linguistic metrics."""
    bleu_score: float
    rouge_scores: Dict[str, float]
    fluency_score: float
    readability_score: float
    diversity_score: float  # Lexical diversity
    factual_consistency: float
    citation_accuracy: float  # Accuracy of citations/references


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    query: str
    ground_truth_answer: Optional[str]
    generated_answer: str
    retrieved_documents: List[Dict[str, Any]]
    relevant_document_ids: List[str]
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    quality_metrics: QualityMetrics
    overall_score: float
    evaluation_time: float
    timestamp: str
    metadata: Dict[str, Any]


class RetrievalEvaluator:
    """Evaluates retrieval performance using standard IR metrics."""
    
    def __init__(self):
        self.config = Config()
    
    def calculate_precision_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """Calculate Precision@K."""
        if k == 0 or not retrieved_ids:
            return 0.0
        
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
        
        return relevant_retrieved / min(k, len(retrieved_at_k))
    
    def calculate_recall_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """Calculate Recall@K."""
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
        
        return relevant_retrieved / len(relevant_ids)
    
    def calculate_f1_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """Calculate F1@K."""
        precision = self.calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        recall = self.calculate_recall_at_k(retrieved_ids, relevant_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_ndcg_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        relevance_scores: Optional[Dict[str, float]], 
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        if k == 0 or not retrieved_ids:
            return 0.0
        
        # Use binary relevance if scores not provided
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG
        sorted_relevant = sorted(
            [(doc_id, score) for doc_id, score in relevance_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        idcg = 0.0
        for i, (_, relevance) in enumerate(sorted_relevant[:k]):
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_hit_rate_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """Calculate Hit Rate@K (whether any relevant document is in top-k)."""
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        return 1.0 if retrieved_at_k & set(relevant_ids) else 0.0
    
    def calculate_average_precision(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """Calculate Average Precision."""
        if not relevant_ids:
            return 0.0
        
        precision_sum = 0.0
        relevant_found = 0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_ids)
    
    def evaluate_retrieval(
        self,
        retrieved_documents: List[Dict[str, Any]],
        relevant_document_ids: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k_values: Optional[List[int]] = None,
        retrieval_time: float = 0.0
    ) -> RetrievalMetrics:
        """Evaluate retrieval performance with comprehensive metrics."""
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        # Extract document IDs
        retrieved_ids = [doc.get('id', str(i)) for i, doc in enumerate(retrieved_documents)]
        
        # Calculate metrics for different k values
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        ndcg_at_k = {}
        hit_rate_at_k = {}
        
        for k in k_values:
            precision_at_k[k] = self.calculate_precision_at_k(retrieved_ids, relevant_document_ids, k)
            recall_at_k[k] = self.calculate_recall_at_k(retrieved_ids, relevant_document_ids, k)
            f1_at_k[k] = self.calculate_f1_at_k(retrieved_ids, relevant_document_ids, k)
            ndcg_at_k[k] = self.calculate_ndcg_at_k(retrieved_ids, relevant_document_ids, relevance_scores, k)
            hit_rate_at_k[k] = self.calculate_hit_rate_at_k(retrieved_ids, relevant_document_ids, k)
        
        # Calculate other metrics
        mrr = self.calculate_mrr(retrieved_ids, relevant_document_ids)
        avg_precision = self.calculate_average_precision(retrieved_ids, relevant_document_ids)
        
        # Calculate coverage
        total_relevant = len(relevant_document_ids)
        retrieved_relevant = len(set(retrieved_ids) & set(relevant_document_ids))
        coverage = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            mean_reciprocal_rank=mrr,
            normalized_dcg_at_k=ndcg_at_k,
            hit_rate_at_k=hit_rate_at_k,
            average_precision=avg_precision,
            retrieval_latency=retrieval_time,
            coverage=coverage
        )


class GenerationEvaluator:
    """Evaluates generation quality using semantic and factual metrics."""
    
    def __init__(self):
        self.config = Config()
        self.sentence_model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logging.warning(f"Failed to load sentence transformer: {e}")
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.sentence_model or not text1.strip() or not text2.strip():
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            return len(words1 & words2) / len(words1 | words2)
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
            return float(similarity)
        except Exception as e:
            logging.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def calculate_faithfulness(self, answer: str, source_texts: List[str]) -> float:
        """
        Calculate faithfulness - how well the answer is grounded in the source texts.
        Higher score means better grounding.
        """
        if not answer.strip() or not source_texts:
            return 0.0
        
        # Extract claims from answer (simplified - sentences)
        if NLTK_AVAILABLE:
            try:
                answer_sentences = sent_tokenize(answer)
            except:
                answer_sentences = answer.split('.')
        else:
            answer_sentences = answer.split('.')
        
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]
        
        if not answer_sentences:
            return 0.0
        
        # Check how many answer sentences are supported by sources
        supported_sentences = 0
        source_text = ' '.join(source_texts)
        
        for sentence in answer_sentences:
            # Simple approach: check semantic similarity with source text
            similarity = self.calculate_semantic_similarity(sentence, source_text)
            if similarity > 0.3:  # Threshold for considering it supported
                supported_sentences += 1
        
        return supported_sentences / len(answer_sentences)
    
    def calculate_answer_relevancy(self, answer: str, question: str) -> float:
        """Calculate how relevant the answer is to the question."""
        return self.calculate_semantic_similarity(answer, question)
    
    def calculate_context_precision(
        self, 
        retrieved_contexts: List[str], 
        relevant_contexts: List[str]
    ) -> float:
        """Calculate precision of retrieved context."""
        if not retrieved_contexts:
            return 0.0
        
        relevant_retrieved = 0
        
        for retrieved in retrieved_contexts:
            for relevant in relevant_contexts:
                if self.calculate_semantic_similarity(retrieved, relevant) > 0.5:
                    relevant_retrieved += 1
                    break
        
        return relevant_retrieved / len(retrieved_contexts)
    
    def calculate_context_recall(
        self, 
        retrieved_contexts: List[str], 
        relevant_contexts: List[str]
    ) -> float:
        """Calculate recall of retrieved context."""
        if not relevant_contexts:
            return 0.0
        
        relevant_retrieved = 0
        
        for relevant in relevant_contexts:
            for retrieved in retrieved_contexts:
                if self.calculate_semantic_similarity(relevant, retrieved) > 0.5:
                    relevant_retrieved += 1
                    break
        
        return relevant_retrieved / len(relevant_contexts)
    
    def calculate_hallucination_score(self, answer: str, source_texts: List[str]) -> float:
        """
        Calculate hallucination score (0-1, where 1 means high hallucination).
        This is essentially 1 - faithfulness.
        """
        faithfulness = self.calculate_faithfulness(answer, source_texts)
        return 1.0 - faithfulness
    
    def calculate_coherence_score(self, text: str) -> float:
        """Calculate internal coherence of text."""
        if not text.strip():
            return 0.0
        
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = text.split('.')
        else:
            sentences = text.split('.')
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent by definition
        
        # Calculate average similarity between consecutive sentences
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            similarity = self.calculate_semantic_similarity(sentences[i], sentences[i + 1])
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def calculate_completeness_score(self, answer: str, question: str, reference_answer: Optional[str] = None) -> float:
        """Calculate how complete the answer is."""
        if not answer.strip():
            return 0.0
        
        # Basic completeness metrics
        completeness_factors = []
        
        # Length factor (longer answers might be more complete, but with diminishing returns)
        answer_length = len(answer.split())
        length_score = min(1.0, answer_length / 50.0)  # Normalize to 50 words
        completeness_factors.append(length_score)
        
        # Question coverage (how well the answer addresses the question)
        question_coverage = self.calculate_semantic_similarity(answer, question)
        completeness_factors.append(question_coverage)
        
        # Reference comparison if available
        if reference_answer:
            reference_similarity = self.calculate_semantic_similarity(answer, reference_answer)
            completeness_factors.append(reference_similarity)
        
        return np.mean(completeness_factors)
    
    def evaluate_generation(
        self,
        question: str,
        generated_answer: str,
        source_documents: List[Dict[str, Any]],
        reference_answer: Optional[str] = None,
        relevant_contexts: Optional[List[str]] = None,
        generation_time: float = 0.0
    ) -> GenerationMetrics:
        """Evaluate generation quality with comprehensive metrics."""
        source_texts = [doc.get('content', '') for doc in source_documents]
        retrieved_contexts = source_texts  # Simplified
        relevant_contexts = relevant_contexts or []
        
        # Calculate all metrics
        faithfulness = self.calculate_faithfulness(generated_answer, source_texts)
        answer_relevancy = self.calculate_answer_relevancy(generated_answer, question)
        context_precision = self.calculate_context_precision(retrieved_contexts, relevant_contexts)
        context_recall = self.calculate_context_recall(retrieved_contexts, relevant_contexts)
        context_relevancy = np.mean([
            self.calculate_semantic_similarity(context, question) 
            for context in retrieved_contexts
        ]) if retrieved_contexts else 0.0
        
        # Overall answer correctness (weighted combination)
        answer_correctness = (
            faithfulness * 0.4 +
            answer_relevancy * 0.3 +
            context_precision * 0.2 +
            context_recall * 0.1
        )
        
        # Semantic similarity to reference
        answer_semantic_similarity = 0.0
        if reference_answer:
            answer_semantic_similarity = self.calculate_semantic_similarity(
                generated_answer, reference_answer
            )
        
        # Other quality metrics
        hallucination_score = self.calculate_hallucination_score(generated_answer, source_texts)
        coherence_score = self.calculate_coherence_score(generated_answer)
        completeness_score = self.calculate_completeness_score(
            generated_answer, question, reference_answer
        )
        
        return GenerationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            context_relevancy=context_relevancy,
            answer_correctness=answer_correctness,
            answer_semantic_similarity=answer_semantic_similarity,
            hallucination_score=hallucination_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            generation_latency=generation_time
        )


class QualityEvaluator:
    """Evaluates text quality and linguistic metrics."""
    
    def __init__(self):
        self.config = Config()
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score."""
        if not NLTK_AVAILABLE or not generated.strip() or not reference.strip():
            return 0.0
        
        try:
            reference_tokens = word_tokenize(reference.lower())
            generated_tokens = word_tokenize(generated.lower())
            
            smoothing = SmoothingFunction().method1
            return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
        except Exception:
            return 0.0
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores (simplified implementation)."""
        if not generated.strip() or not reference.strip():
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        
        # Simple n-gram overlap calculation
        def get_ngrams(text: str, n: int) -> set:
            if NLTK_AVAILABLE:
                try:
                    tokens = word_tokenize(text.lower())
                except:
                    tokens = text.lower().split()
            else:
                tokens = text.lower().split()
            
            return set(zip(*[tokens[i:] for i in range(n)]))
        
        # ROUGE-1 (unigram overlap)
        gen_unigrams = get_ngrams(generated, 1)
        ref_unigrams = get_ngrams(reference, 1)
        
        rouge_1 = len(gen_unigrams & ref_unigrams) / len(ref_unigrams) if ref_unigrams else 0.0
        
        # ROUGE-2 (bigram overlap)
        gen_bigrams = get_ngrams(generated, 2)
        ref_bigrams = get_ngrams(reference, 2)
        
        rouge_2 = len(gen_bigrams & ref_bigrams) / len(ref_bigrams) if ref_bigrams else 0.0
        
        # ROUGE-L (simplified as unigram F1)
        if gen_unigrams and ref_unigrams:
            overlap = len(gen_unigrams & ref_unigrams)
            precision = overlap / len(gen_unigrams)
            recall = overlap / len(ref_unigrams)
            rouge_l = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            rouge_l = 0.0
        
        return {
            "rouge-1": rouge_1,
            "rouge-2": rouge_2,
            "rouge-l": rouge_l
        }
    
    def calculate_fluency_score(self, text: str) -> float:
        """Calculate fluency score based on linguistic features."""
        if not text.strip():
            return 0.0
        
        # Simple fluency indicators
        fluency_factors = []
        
        # Sentence structure (avoid very short or very long sentences)
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = text.split('.')
        else:
            sentences = text.split('.')
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            # Optimal range: 10-25 words per sentence
            length_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
            length_score = max(0.0, min(1.0, length_score))
            fluency_factors.append(length_score)
        
        # Vocabulary diversity
        words = text.lower().split()
        if words:
            unique_words = len(set(words))
            diversity = unique_words / len(words)
            fluency_factors.append(diversity)
        
        # Repetition penalty
        if words and len(words) > 1:
            word_counts = Counter(words)
            max_repetition = max(word_counts.values())
            repetition_penalty = 1.0 - (max_repetition - 1) / len(words)
            fluency_factors.append(max(0.0, repetition_penalty))
        
        return np.mean(fluency_factors) if fluency_factors else 0.0
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)."""
        if not text.strip():
            return 0.0
        
        # Count sentences and words
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                words = word_tokenize(text)
            except:
                sentences = text.split('.')
                words = text.split()
        else:
            sentences = text.split('.')
            words = text.split()
        
        sentences = [s.strip() for s in sentences if s.strip()]
        words = [w for w in words if w.isalpha()]
        
        if not sentences or not words:
            return 0.0
        
        # Approximate syllable count
        def count_syllables(word: str) -> int:
            word = word.lower()
            vowels = "aeiouy"
            syllable_count = 0
            previous_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = is_vowel
            
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            return max(1, syllable_count)
        
        total_syllables = sum(count_syllables(word) for word in words)
        
        # Flesch Reading Ease
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, flesch_score / 100.0))
    
    def evaluate_quality(
        self,
        generated_text: str,
        reference_text: Optional[str] = None
    ) -> QualityMetrics:
        """Evaluate text quality with linguistic metrics."""
        bleu_score = 0.0
        rouge_scores = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        
        if reference_text:
            bleu_score = self.calculate_bleu_score(generated_text, reference_text)
            rouge_scores = self.calculate_rouge_scores(generated_text, reference_text)
        
        fluency_score = self.calculate_fluency_score(generated_text)
        readability_score = self.calculate_readability_score(generated_text)
        
        # Diversity score (lexical diversity)
        words = generated_text.lower().split()
        diversity_score = len(set(words)) / len(words) if words else 0.0
        
        # Placeholder scores for advanced metrics
        factual_consistency = 0.8  # Would need fact-checking
        citation_accuracy = 0.9   # Would need citation analysis
        
        return QualityMetrics(
            bleu_score=bleu_score,
            rouge_scores=rouge_scores,
            fluency_score=fluency_score,
            readability_score=readability_score,
            diversity_score=diversity_score,
            factual_consistency=factual_consistency,
            citation_accuracy=citation_accuracy
        )


class RAGEvaluator:
    """Comprehensive RAG system evaluator."""
    
    def __init__(self):
        self.config = Config()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.quality_evaluator = QualityEvaluator()
        
        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []
        
    def evaluate_single_query(
        self,
        query: str,
        generated_answer: str,
        retrieved_documents: List[Dict[str, Any]],
        relevant_document_ids: List[str],
        ground_truth_answer: Optional[str] = None,
        relevance_scores: Optional[Dict[str, float]] = None,
        retrieval_time: float = 0.0,
        generation_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate a single query-answer pair comprehensively."""
        start_time = time.time()
        
        # Evaluate retrieval
        retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
            retrieved_documents=retrieved_documents,
            relevant_document_ids=relevant_document_ids,
            relevance_scores=relevance_scores,
            retrieval_time=retrieval_time
        )
        
        # Evaluate generation
        generation_metrics = self.generation_evaluator.evaluate_generation(
            question=query,
            generated_answer=generated_answer,
            source_documents=retrieved_documents,
            reference_answer=ground_truth_answer,
            generation_time=generation_time
        )
        
        # Evaluate quality
        quality_metrics = self.quality_evaluator.evaluate_quality(
            generated_text=generated_answer,
            reference_text=ground_truth_answer
        )
        
        # Calculate overall score (weighted combination)
        overall_score = (
            np.mean(list(retrieval_metrics.precision_at_k.values())) * 0.3 +
            generation_metrics.answer_correctness * 0.4 +
            quality_metrics.fluency_score * 0.2 +
            (1.0 - generation_metrics.hallucination_score) * 0.1
        )
        
        evaluation_time = time.time() - start_time
        
        result = EvaluationResult(
            query=query,
            ground_truth_answer=ground_truth_answer,
            generated_answer=generated_answer,
            retrieved_documents=retrieved_documents,
            relevant_document_ids=relevant_document_ids,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            quality_metrics=quality_metrics,
            overall_score=overall_score,
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Add to history
        self.evaluation_history.append(result)
        
        return result
    
    def evaluate_dataset(
        self,
        evaluation_dataset: List[Dict[str, Any]],
        rag_system_callable: Callable[[str], Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system on a complete dataset.
        
        Args:
            evaluation_dataset: List of evaluation examples with required fields
            rag_system_callable: Function that takes a query and returns RAG results
            progress_callback: Optional callback for progress updates
        
        Expected dataset format:
        [
            {
                "query": "What is...",
                "ground_truth_answer": "The answer is...",
                "relevant_document_ids": ["doc1", "doc2"],
                "relevance_scores": {"doc1": 1.0, "doc2": 0.8}  # optional
            },
            ...
        ]
        """
        results = []
        start_time = time.time()
        
        for i, example in enumerate(evaluation_dataset):
            try:
                # Get RAG system response
                rag_response = rag_system_callable(example["query"])
                
                # Extract information from response
                generated_answer = rag_response.get("answer", "")
                retrieved_docs = rag_response.get("sources", [])
                retrieval_time = rag_response.get("retrieval_time", 0.0)
                generation_time = rag_response.get("response_time", 0.0) - retrieval_time
                
                # Evaluate this example
                result = self.evaluate_single_query(
                    query=example["query"],
                    generated_answer=generated_answer,
                    retrieved_documents=retrieved_docs,
                    relevant_document_ids=example["relevant_document_ids"],
                    ground_truth_answer=example.get("ground_truth_answer"),
                    relevance_scores=example.get("relevance_scores"),
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    metadata={"dataset_index": i}
                )
                
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(evaluation_dataset))
                    
            except Exception as e:
                logging.error(f"Evaluation failed for example {i}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            "individual_results": results,
            "aggregate_metrics": aggregate_metrics,
            "dataset_size": len(evaluation_dataset),
            "successful_evaluations": len(results),
            "total_evaluation_time": total_time,
            "average_time_per_query": total_time / len(results) if results else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate aggregate metrics across multiple evaluation results."""
        if not results:
            return {}
        
        # Aggregate retrieval metrics
        all_precision_at_k = defaultdict(list)
        all_recall_at_k = defaultdict(list)
        all_f1_at_k = defaultdict(list)
        all_ndcg_at_k = defaultdict(list)
        all_hit_rate_at_k = defaultdict(list)
        
        for result in results:
            for k, value in result.retrieval_metrics.precision_at_k.items():
                all_precision_at_k[k].append(value)
            for k, value in result.retrieval_metrics.recall_at_k.items():
                all_recall_at_k[k].append(value)
            for k, value in result.retrieval_metrics.f1_at_k.items():
                all_f1_at_k[k].append(value)
            for k, value in result.retrieval_metrics.normalized_dcg_at_k.items():
                all_ndcg_at_k[k].append(value)
            for k, value in result.retrieval_metrics.hit_rate_at_k.items():
                all_hit_rate_at_k[k].append(value)
        
        # Aggregate generation metrics
        generation_attrs = [
            'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall',
            'context_relevancy', 'answer_correctness', 'answer_semantic_similarity',
            'hallucination_score', 'coherence_score', 'completeness_score'
        ]
        
        # Aggregate quality metrics
        quality_attrs = [
            'bleu_score', 'fluency_score', 'readability_score', 'diversity_score',
            'factual_consistency', 'citation_accuracy'
        ]
        
        aggregate = {
            "retrieval_metrics": {
                "precision_at_k": {k: np.mean(values) for k, values in all_precision_at_k.items()},
                "recall_at_k": {k: np.mean(values) for k, values in all_recall_at_k.items()},
                "f1_at_k": {k: np.mean(values) for k, values in all_f1_at_k.items()},
                "ndcg_at_k": {k: np.mean(values) for k, values in all_ndcg_at_k.items()},
                "hit_rate_at_k": {k: np.mean(values) for k, values in all_hit_rate_at_k.items()},
                "mean_reciprocal_rank": np.mean([r.retrieval_metrics.mean_reciprocal_rank for r in results]),
                "average_precision": np.mean([r.retrieval_metrics.average_precision for r in results]),
                "coverage": np.mean([r.retrieval_metrics.coverage for r in results]),
                "retrieval_latency": np.mean([r.retrieval_metrics.retrieval_latency for r in results])
            },
            "generation_metrics": {
                attr: np.mean([getattr(r.generation_metrics, attr) for r in results])
                for attr in generation_attrs
            },
            "quality_metrics": {
                attr: np.mean([getattr(r.quality_metrics, attr) for r in results])
                for attr in quality_attrs if attr != 'rouge_scores'
            },
            "overall_score": np.mean([r.overall_score for r in results]),
            "score_distribution": {
                "min": min([r.overall_score for r in results]),
                "max": max([r.overall_score for r in results]),
                "std": np.std([r.overall_score for r in results]),
                "median": np.median([r.overall_score for r in results])
            }
        }
        
        # Add ROUGE scores separately
        rouge_1_scores = [r.quality_metrics.rouge_scores.get('rouge-1', 0.0) for r in results]
        rouge_2_scores = [r.quality_metrics.rouge_scores.get('rouge-2', 0.0) for r in results]
        rouge_l_scores = [r.quality_metrics.rouge_scores.get('rouge-l', 0.0) for r in results]
        
        aggregate["quality_metrics"]["rouge_scores"] = {
            "rouge-1": np.mean(rouge_1_scores),
            "rouge-2": np.mean(rouge_2_scores),
            "rouge-l": np.mean(rouge_l_scores)
        }
        
        return aggregate
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """Export evaluation results to file."""
        if format.lower() == 'json':
            # Convert dataclasses to dictionaries
            export_data = []
            for result in self.evaluation_history:
                export_data.append({
                    "query": result.query,
                    "ground_truth_answer": result.ground_truth_answer,
                    "generated_answer": result.generated_answer,
                    "retrieval_metrics": asdict(result.retrieval_metrics),
                    "generation_metrics": asdict(result.generation_metrics),
                    "quality_metrics": asdict(result.quality_metrics),
                    "overall_score": result.overall_score,
                    "evaluation_time": result.evaluation_time,
                    "timestamp": result.timestamp,
                    "metadata": result.metadata
                })
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        print(f"📊 Exported {len(self.evaluation_history)} evaluation results to {filepath}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all evaluations."""
        if not self.evaluation_history:
            return {"error": "No evaluation results available"}
        
        aggregate_metrics = self.calculate_aggregate_metrics(self.evaluation_history)
        
        # Performance trends
        recent_results = self.evaluation_history[-10:] if len(self.evaluation_history) >= 10 else self.evaluation_history
        recent_scores = [r.overall_score for r in recent_results]
        
        return {
            "summary": {
                "total_evaluations": len(self.evaluation_history),
                "average_overall_score": aggregate_metrics["overall_score"],
                "recent_performance_trend": {
                    "recent_average": np.mean(recent_scores),
                    "improvement": np.mean(recent_scores) - aggregate_metrics["overall_score"]
                }
            },
            "top_performing_queries": [
                {
                    "query": r.query,
                    "score": r.overall_score,
                    "timestamp": r.timestamp
                }
                for r in sorted(self.evaluation_history, key=lambda x: x.overall_score, reverse=True)[:5]
            ],
            "lowest_performing_queries": [
                {
                    "query": r.query,
                    "score": r.overall_score,
                    "timestamp": r.timestamp
                }
                for r in sorted(self.evaluation_history, key=lambda x: x.overall_score)[:5]
            ],
            "aggregate_metrics": aggregate_metrics
        }
    
    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.evaluation_history.clear()
        print("🧹 Evaluation history cleared")


def create_sample_evaluation_dataset() -> List[Dict[str, Any]]:
    """Create a sample evaluation dataset for testing."""
    return [
        {
            "query": "What is machine learning?",
            "ground_truth_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "relevant_document_ids": ["doc_ml_intro", "doc_ai_overview"],
            "relevance_scores": {"doc_ml_intro": 1.0, "doc_ai_overview": 0.7}
        },
        {
            "query": "How does natural language processing work?",
            "ground_truth_answer": "Natural language processing combines computational linguistics with machine learning to help computers understand, interpret, and generate human language.",
            "relevant_document_ids": ["doc_nlp_basics", "doc_language_models"],
            "relevance_scores": {"doc_nlp_basics": 1.0, "doc_language_models": 0.9}
        },
        {
            "query": "What are the applications of deep learning?",
            "ground_truth_answer": "Deep learning has applications in computer vision, natural language processing, speech recognition, autonomous vehicles, and medical diagnosis.",
            "relevant_document_ids": ["doc_deep_learning", "doc_dl_applications"],
            "relevance_scores": {"doc_deep_learning": 0.8, "doc_dl_applications": 1.0}
        }
    ]