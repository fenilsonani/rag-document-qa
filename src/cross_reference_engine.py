"""
Multi-Document Cross-Referencing and Relationship Mapping Engine
Advanced system for finding connections, contradictions, and relationships across documents.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from datetime import datetime

from langchain.schema import Document
from .config import Config


class CrossReferenceEngine:
    """Advanced multi-document cross-referencing and relationship mapping."""
    
    def __init__(self):
        self.config = Config()
        self.similarity_threshold = 0.3
        self.contradiction_keywords = [
            'however', 'but', 'although', 'despite', 'nevertheless', 'on the contrary',
            'in contrast', 'whereas', 'while', 'conversely', 'differently', 'oppose'
        ]
        self.support_keywords = [
            'similarly', 'likewise', 'furthermore', 'moreover', 'additionally',
            'also', 'agrees', 'confirms', 'supports', 'consistent', 'accordance'
        ]
    
    def analyze_document_relationships(self, documents: List[Document]) -> Dict[str, Any]:
        """Comprehensive analysis of relationships between documents."""
        if len(documents) < 2:
            return {"error": "Need at least 2 documents for cross-referencing"}
        
        # Create document fingerprints
        doc_fingerprints = self._create_document_fingerprints(documents)
        
        # Find semantic similarities
        similarities = self._calculate_document_similarities(documents)
        
        # Detect contradictions and agreements
        contradictions = self._detect_contradictions(documents)
        agreements = self._detect_agreements(documents)
        
        # Build relationship graph
        relationship_graph = self._build_relationship_graph(documents, similarities, contradictions, agreements)
        
        # Find citation patterns and references
        citations = self._detect_citations_and_references(documents)
        
        # Analyze temporal relationships
        temporal_analysis = self._analyze_temporal_relationships(documents)
        
        # Generate insights
        insights = self._generate_relationship_insights(
            similarities, contradictions, agreements, relationship_graph
        )
        
        return {
            "document_fingerprints": doc_fingerprints,
            "similarity_matrix": similarities,
            "contradictions": contradictions,
            "agreements": agreements,
            "relationship_graph": relationship_graph,
            "citations": citations,
            "temporal_analysis": temporal_analysis,
            "insights": insights,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _create_document_fingerprints(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Create unique fingerprints for each document."""
        fingerprints = []
        
        for i, doc in enumerate(documents):
            content = doc.page_content
            words = content.lower().split()
            
            # Extract key characteristics
            fingerprint = {
                "doc_id": i,
                "filename": doc.metadata.get('filename', f'Document_{i}'),
                "word_count": len(words),
                "unique_words": len(set(words)),
                "avg_sentence_length": self._calculate_avg_sentence_length(content),
                "key_terms": self._extract_key_terms(content),
                "document_type": self._classify_document_type(content),
                "formality_score": self._calculate_formality_score(content),
                "technical_density": self._calculate_technical_density(content)
            }
            
            fingerprints.append(fingerprint)
        
        return fingerprints
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        return sum(len(s.split()) for s in sentences) / len(sentences)
    
    def _extract_key_terms(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key terms using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[i] for i in top_indices if scores[i] > 0]
        except:
            # Fallback to word frequency
            words = text.lower().split()
            word_freq = Counter(w for w in words if len(w) > 3 and w.isalpha())
            return [word for word, freq in word_freq.most_common(top_n)]
    
    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on content characteristics."""
        text_lower = text.lower()
        
        # Academic indicators
        academic_indicators = ['abstract', 'methodology', 'conclusion', 'references', 'hypothesis', 'analysis']
        academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)
        
        # Technical indicators
        technical_indicators = ['algorithm', 'implementation', 'system', 'method', 'performance', 'data']
        technical_score = sum(1 for indicator in technical_indicators if indicator in text_lower)
        
        # Business indicators
        business_indicators = ['market', 'revenue', 'profit', 'customer', 'strategy', 'business']
        business_score = sum(1 for indicator in business_indicators if indicator in text_lower)
        
        # Legal indicators
        legal_indicators = ['shall', 'pursuant', 'whereas', 'therefore', 'contract', 'agreement']
        legal_score = sum(1 for indicator in legal_indicators if indicator in text_lower)
        
        scores = {
            'academic': academic_score,
            'technical': technical_score,
            'business': business_score,
            'legal': legal_score
        }
        
        max_type = max(scores, key=scores.get)
        return max_type if scores[max_type] > 2 else 'general'
    
    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score of the document."""
        formal_indicators = [
            'furthermore', 'moreover', 'nevertheless', 'consequently', 'therefore',
            'additionally', 'specifically', 'particularly', 'subsequently'
        ]
        informal_indicators = [
            "it's", "don't", "can't", "won't", "I'm", "you're", "we're",
            'really', 'pretty', 'quite', 'very', 'totally'
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.5
        
        formal_ratio = formal_count / total_words * 1000
        informal_ratio = informal_count / total_words * 1000
        
        # Scale to 0-1 where 1 is very formal
        return min(1.0, max(0.0, (formal_ratio - informal_ratio + 5) / 10))
    
    def _calculate_technical_density(self, text: str) -> float:
        """Calculate technical density of the document."""
        technical_patterns = [
            r'\b\w+\(\)',  # Function calls
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\d+\.?\d*%\b',  # Percentages
            r'\b\d+\.?\d*[a-zA-Z]+\b',  # Numbers with units
            r'\b[a-zA-Z]+\d+\b',  # Alphanumeric codes
        ]
        
        technical_matches = 0
        for pattern in technical_patterns:
            technical_matches += len(re.findall(pattern, text))
        
        total_words = len(text.split())
        return min(1.0, technical_matches / max(total_words, 1) * 100)
    
    def _calculate_document_similarities(self, documents: List[Document]) -> Dict[str, Any]:
        """Calculate semantic similarities between all document pairs."""
        texts = [doc.page_content for doc in documents]
        
        try:
            # Use TF-IDF for similarity calculation
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create similarity pairs
            similarity_pairs = []
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    similarity_score = similarity_matrix[i][j]
                    if similarity_score > self.similarity_threshold:
                        similarity_pairs.append({
                            "doc1_id": i,
                            "doc2_id": j,
                            "doc1_name": documents[i].metadata.get('filename', f'Document_{i}'),
                            "doc2_name": documents[j].metadata.get('filename', f'Document_{j}'),
                            "similarity_score": round(float(similarity_score), 3),
                            "relationship_strength": self._classify_similarity_strength(similarity_score)
                        })
            
            return {
                "similarity_matrix": similarity_matrix.tolist(),
                "high_similarity_pairs": sorted(similarity_pairs, key=lambda x: x['similarity_score'], reverse=True),
                "average_similarity": float(np.mean(similarity_matrix[np.triu_indices(len(documents), k=1)]))
            }
        
        except Exception as e:
            return {"error": f"Similarity calculation failed: {str(e)}"}
    
    def _classify_similarity_strength(self, score: float) -> str:
        """Classify similarity strength."""
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Moderate"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _detect_contradictions(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Detect potential contradictions between documents."""
        contradictions = []
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc1 = documents[i]
                doc2 = documents[j]
                
                # Find contradictory statements
                contradictory_pairs = self._find_contradictory_statements(
                    doc1.page_content, doc2.page_content
                )
                
                if contradictory_pairs:
                    contradictions.append({
                        "doc1_id": i,
                        "doc2_id": j,
                        "doc1_name": doc1.metadata.get('filename', f'Document_{i}'),
                        "doc2_name": doc2.metadata.get('filename', f'Document_{j}'),
                        "contradictory_statements": contradictory_pairs,
                        "contradiction_count": len(contradictory_pairs)
                    })
        
        return sorted(contradictions, key=lambda x: x['contradiction_count'], reverse=True)
    
    def _find_contradictory_statements(self, text1: str, text2: str) -> List[Dict[str, str]]:
        """Find specific contradictory statements between two texts."""
        contradictory_pairs = []
        
        # Split into sentences
        sentences1 = re.split(r'[.!?]+', text1)
        sentences2 = re.split(r'[.!?]+', text2)
        
        sentences1 = [s.strip() for s in sentences1 if len(s.strip()) > 10]
        sentences2 = [s.strip() for s in sentences2 if len(s.strip()) > 10]
        
        # Look for opposing statements
        negation_patterns = [
            (r'\bis\s+', r'\bis\s+not\s+'),
            (r'\bwill\s+', r'\bwill\s+not\s+'),
            (r'\bcan\s+', r'\bcannot\s+'),
            (r'\bshould\s+', r'\bshould\s+not\s+'),
            (r'\bincrease', r'\bdecrease'),
            (r'\bhigh', r'\blow'),
            (r'\bpositive', r'\bnegative'),
            (r'\beffective', r'\bineffective'),
            (r'\bsupport', r'\boppose'),
        ]
        
        for sent1 in sentences1[:20]:  # Limit for performance
            for sent2 in sentences2[:20]:
                # Check for direct negations
                for positive_pattern, negative_pattern in negation_patterns:
                    if (re.search(positive_pattern, sent1, re.IGNORECASE) and 
                        re.search(negative_pattern, sent2, re.IGNORECASE)) or \
                       (re.search(negative_pattern, sent1, re.IGNORECASE) and 
                        re.search(positive_pattern, sent2, re.IGNORECASE)):
                        
                        # Check if they're talking about similar topics
                        common_words = set(sent1.lower().split()) & set(sent2.lower().split())
                        if len(common_words) >= 3:
                            contradictory_pairs.append({
                                "statement1": sent1[:200] + "..." if len(sent1) > 200 else sent1,
                                "statement2": sent2[:200] + "..." if len(sent2) > 200 else sent2,
                                "contradiction_type": "negation",
                                "confidence": min(1.0, len(common_words) / 10)
                            })
        
        return contradictory_pairs[:5]  # Limit results
    
    def _detect_agreements(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Detect agreements and supporting statements between documents."""
        agreements = []
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc1 = documents[i]
                doc2 = documents[j]
                
                # Find supporting statements
                supporting_pairs = self._find_supporting_statements(
                    doc1.page_content, doc2.page_content
                )
                
                if supporting_pairs:
                    agreements.append({
                        "doc1_id": i,
                        "doc2_id": j,
                        "doc1_name": doc1.metadata.get('filename', f'Document_{i}'),
                        "doc2_name": doc2.metadata.get('filename', f'Document_{j}'),
                        "supporting_statements": supporting_pairs,
                        "agreement_count": len(supporting_pairs)
                    })
        
        return sorted(agreements, key=lambda x: x['agreement_count'], reverse=True)
    
    def _find_supporting_statements(self, text1: str, text2: str) -> List[Dict[str, str]]:
        """Find supporting statements between two texts."""
        supporting_pairs = []
        
        sentences1 = re.split(r'[.!?]+', text1)
        sentences2 = re.split(r'[.!?]+', text2)
        
        sentences1 = [s.strip() for s in sentences1 if len(s.strip()) > 10]
        sentences2 = [s.strip() for s in sentences2 if len(s.strip()) > 10]
        
        # Use TF-IDF to find similar sentences
        try:
            all_sentences = sentences1 + sentences2
            if len(all_sentences) < 2:
                return supporting_pairs
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(all_sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find high similarity pairs between documents
            for i, sent1 in enumerate(sentences1):
                for j, sent2 in enumerate(sentences2):
                    similarity = similarity_matrix[i][len(sentences1) + j]
                    
                    if similarity > 0.5:  # High similarity threshold
                        # Check for supporting keywords
                        combined_text = (sent1 + " " + sent2).lower()
                        support_score = sum(1 for keyword in self.support_keywords if keyword in combined_text)
                        
                        if support_score > 0 or similarity > 0.7:
                            supporting_pairs.append({
                                "statement1": sent1[:200] + "..." if len(sent1) > 200 else sent1,
                                "statement2": sent2[:200] + "..." if len(sent2) > 200 else sent2,
                                "agreement_type": "semantic_similarity",
                                "confidence": round(float(similarity), 3),
                                "support_keywords": support_score
                            })
        
        except Exception as e:
            # Fallback: simple keyword matching
            for sent1 in sentences1[:10]:
                for sent2 in sentences2[:10]:
                    common_words = set(sent1.lower().split()) & set(sent2.lower().split())
                    if len(common_words) >= 5:
                        supporting_pairs.append({
                            "statement1": sent1[:200] + "..." if len(sent1) > 200 else sent1,
                            "statement2": sent2[:200] + "..." if len(sent2) > 200 else sent2,
                            "agreement_type": "keyword_overlap",
                            "confidence": min(1.0, len(common_words) / 10)
                        })
        
        return supporting_pairs[:5]  # Limit results
    
    def _build_relationship_graph(self, documents: List[Document], similarities: Dict, 
                                 contradictions: List, agreements: List) -> Dict[str, Any]:
        """Build a relationship graph between documents."""
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (documents)
        for i, doc in enumerate(documents):
            G.add_node(i, 
                      name=doc.metadata.get('filename', f'Document_{i}'),
                      word_count=len(doc.page_content.split()))
        
        # Add edges based on relationships
        edge_data = []
        
        # Similarity edges
        for pair in similarities.get('high_similarity_pairs', []):
            G.add_edge(pair['doc1_id'], pair['doc2_id'], 
                      relationship='similarity',
                      weight=pair['similarity_score'])
            edge_data.append({
                "source": pair['doc1_id'],
                "target": pair['doc2_id'],
                "type": "similarity",
                "weight": pair['similarity_score']
            })
        
        # Contradiction edges
        for contradiction in contradictions:
            if G.has_edge(contradiction['doc1_id'], contradiction['doc2_id']):
                G[contradiction['doc1_id']][contradiction['doc2_id']]['contradiction_count'] = contradiction['contradiction_count']
            else:
                G.add_edge(contradiction['doc1_id'], contradiction['doc2_id'],
                          relationship='contradiction',
                          weight=contradiction['contradiction_count'] * 0.1)
            
            edge_data.append({
                "source": contradiction['doc1_id'],
                "target": contradiction['doc2_id'],
                "type": "contradiction",
                "weight": contradiction['contradiction_count']
            })
        
        # Agreement edges
        for agreement in agreements:
            if G.has_edge(agreement['doc1_id'], agreement['doc2_id']):
                G[agreement['doc1_id']][agreement['doc2_id']]['agreement_count'] = agreement['agreement_count']
            else:
                G.add_edge(agreement['doc1_id'], agreement['doc2_id'],
                          relationship='agreement',
                          weight=agreement['agreement_count'] * 0.1)
            
            edge_data.append({
                "source": agreement['doc1_id'],
                "target": agreement['doc2_id'],
                "type": "agreement",
                "weight": agreement['agreement_count']
            })
        
        # Calculate network metrics
        try:
            centrality = nx.degree_centrality(G)
            clustering = nx.clustering(G)
            
            # Find most connected documents
            most_connected = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "nodes": [{"id": i, "name": documents[i].metadata.get('filename', f'Document_{i}')} 
                         for i in range(len(documents))],
                "edges": edge_data,
                "centrality_scores": centrality,
                "clustering_coefficients": clustering,
                "most_connected_documents": most_connected[:3],
                "network_density": nx.density(G),
                "total_relationships": len(edge_data)
            }
        
        except Exception as e:
            return {
                "nodes": [{"id": i, "name": documents[i].metadata.get('filename', f'Document_{i}')} 
                         for i in range(len(documents))],
                "edges": edge_data,
                "error": f"Network analysis failed: {str(e)}"
            }
    
    def _detect_citations_and_references(self, documents: List[Document]) -> Dict[str, List[Dict]]:
        """Detect citations and references between documents."""
        citations = {
            "internal_references": [],
            "external_citations": [],
            "figure_references": [],
            "table_references": []
        }
        
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([^)]*\d{4}[^)]*)\)',  # (Author, 2020)
            r'(?:Figure|Fig\.)\s+(\d+)',  # Figure 1, Fig. 2
            r'(?:Table|Tbl\.)\s+(\d+)',  # Table 1, Tbl. 2
            r'(?:see|See)\s+(?:section|Section)\s+(\d+(?:\.\d+)*)',  # See section 2.1
        ]
        
        for i, doc in enumerate(documents):
            content = doc.page_content
            
            for pattern in citation_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if 'Figure' in pattern or 'Fig' in pattern:
                        citations["figure_references"].append({
                            "document_id": i,
                            "document_name": doc.metadata.get('filename', f'Document_{i}'),
                            "reference": f"Figure {match}",
                            "context": self._extract_context(content, f"Figure {match}")
                        })
                    elif 'Table' in pattern or 'Tbl' in pattern:
                        citations["table_references"].append({
                            "document_id": i,
                            "document_name": doc.metadata.get('filename', f'Document_{i}'),
                            "reference": f"Table {match}",
                            "context": self._extract_context(content, f"Table {match}")
                        })
                    elif '[' in pattern:
                        citations["external_citations"].append({
                            "document_id": i,
                            "document_name": doc.metadata.get('filename', f'Document_{i}'),
                            "citation_number": match,
                            "context": self._extract_context(content, f"[{match}]")
                        })
        
        return citations
    
    def _extract_context(self, text: str, reference: str, context_length: int = 100) -> str:
        """Extract context around a reference."""
        index = text.find(reference)
        if index == -1:
            return ""
        
        start = max(0, index - context_length)
        end = min(len(text), index + len(reference) + context_length)
        
        return "..." + text[start:end] + "..."
    
    def _analyze_temporal_relationships(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze temporal relationships between documents."""
        temporal_info = []
        
        date_pattern = r'\b\d{4}\b'  # Simple year extraction
        
        for i, doc in enumerate(documents):
            years = re.findall(date_pattern, doc.page_content)
            years = [int(year) for year in years if 1900 <= int(year) <= 2030]
            
            if years:
                temporal_info.append({
                    "document_id": i,
                    "document_name": doc.metadata.get('filename', f'Document_{i}'),
                    "years_mentioned": sorted(set(years)),
                    "earliest_year": min(years),
                    "latest_year": max(years),
                    "temporal_span": max(years) - min(years) if years else 0
                })
        
        # Find temporal overlaps
        overlaps = []
        for i in range(len(temporal_info)):
            for j in range(i + 1, len(temporal_info)):
                doc1_years = set(temporal_info[i]["years_mentioned"])
                doc2_years = set(temporal_info[j]["years_mentioned"])
                
                common_years = doc1_years & doc2_years
                if common_years:
                    overlaps.append({
                        "doc1_id": temporal_info[i]["document_id"],
                        "doc2_id": temporal_info[j]["document_id"],
                        "doc1_name": temporal_info[i]["document_name"],
                        "doc2_name": temporal_info[j]["document_name"],
                        "common_years": sorted(list(common_years)),
                        "overlap_strength": len(common_years)
                    })
        
        return {
            "document_temporal_info": temporal_info,
            "temporal_overlaps": sorted(overlaps, key=lambda x: x["overlap_strength"], reverse=True),
            "overall_time_span": {
                "earliest": min([info["earliest_year"] for info in temporal_info]) if temporal_info else None,
                "latest": max([info["latest_year"] for info in temporal_info]) if temporal_info else None
            }
        }
    
    def _generate_relationship_insights(self, similarities: Dict, contradictions: List, 
                                       agreements: List, graph: Dict) -> Dict[str, Any]:
        """Generate insights from relationship analysis."""
        insights = {
            "summary": "",
            "key_findings": [],
            "recommendations": [],
            "relationship_strength": "weak"
        }
        
        # Analyze similarity patterns
        high_sim_count = len(similarities.get('high_similarity_pairs', []))
        avg_similarity = similarities.get('average_similarity', 0)
        
        # Analyze contradictions and agreements
        contradiction_count = len(contradictions)
        agreement_count = len(agreements)
        
        # Generate key findings
        if high_sim_count > 0:
            insights["key_findings"].append(f"Found {high_sim_count} highly similar document pairs")
        
        if contradiction_count > 0:
            insights["key_findings"].append(f"Detected {contradiction_count} potential contradictions")
            insights["recommendations"].append("Review contradictory statements for consistency")
        
        if agreement_count > 0:
            insights["key_findings"].append(f"Identified {agreement_count} supporting relationships")
        
        # Determine relationship strength
        total_relationships = graph.get('total_relationships', 0)
        total_possible = len(similarities.get('similarity_matrix', [[]])) * (len(similarities.get('similarity_matrix', [[]])) - 1) / 2
        
        if total_possible > 0:
            relationship_density = total_relationships / total_possible
            if relationship_density > 0.6:
                insights["relationship_strength"] = "very strong"
            elif relationship_density > 0.4:
                insights["relationship_strength"] = "strong"
            elif relationship_density > 0.2:
                insights["relationship_strength"] = "moderate"
            else:
                insights["relationship_strength"] = "weak"
        
        # Generate summary
        insights["summary"] = f"Analysis of {len(similarities.get('similarity_matrix', []))} documents reveals {insights['relationship_strength']} interconnections with {total_relationships} identified relationships."
        
        # Generate recommendations
        if avg_similarity < 0.3:
            insights["recommendations"].append("Documents appear to cover diverse topics - consider organizing by theme")
        
        if contradiction_count > agreement_count:
            insights["recommendations"].append("High contradiction rate - review for consistency")
        
        if not insights["recommendations"]:
            insights["recommendations"].append("Document relationships appear well-balanced")
        
        return insights