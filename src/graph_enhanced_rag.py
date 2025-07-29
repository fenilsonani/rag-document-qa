"""
Graph-Enhanced RAG System - Pro-Level Enhancement
Implements intelligent knowledge graph construction and graph-based retrieval for enhanced reasoning.
"""

import re
import json
import logging
import hashlib
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime
import pickle
from pathlib import Path

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        SPACY_AVAILABLE = False
        logging.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Named entity recognition will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Graph embeddings will be limited.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Advanced graph analysis will be limited.")

from langchain.schema import Document
from .config import Config


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    entity_id: str
    name: str
    entity_type: str  # PERSON, ORG, LOCATION, CONCEPT, etc.
    aliases: Set[str]
    attributes: Dict[str, Any]
    mentions: List[Dict[str, Any]]  # Document mentions
    confidence: float
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['aliases'] = list(result['aliases'])
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result


@dataclass
class Relation:
    """Represents a relation between entities."""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    evidence: List[str]  # Supporting text snippets
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class KnowledgeTriple:
    """Represents a knowledge triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_document: str
    context: str


@dataclass
class GraphQuery:
    """Represents a graph query with semantic understanding."""
    original_query: str
    entities_mentioned: List[str]
    relations_queried: List[str]
    query_type: str  # factual, relational, exploratory, etc.
    semantic_expansion: List[str]


class EntityExtractor:
    """Advanced entity extraction using multiple approaches."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.entity_patterns = self._load_entity_patterns()
        self.sentence_model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for entity extraction."""
        return {
            'EMAIL': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'PHONE': [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'],
            'URL': [r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'],
            'DATE': [r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'],
            'MONEY': [r'\$\d+(?:,\d{3})*(?:\.\d{2})?', r'\b\d+(?:,\d{3})*\s*(?:dollars?|USD|euros?|EUR)\b'],
            'PERCENTAGE': [r'\b\d+(?:\.\d+)?%\b'],
        }
    
    def extract_entities(self, text: str, document_id: str = None) -> List[Entity]:
        """Extract entities from text using multiple methods."""
        entities = []
        
        # Method 1: spaCy NER
        if SPACY_AVAILABLE:
            spacy_entities = self._extract_spacy_entities(text, document_id)
            entities.extend(spacy_entities)
        
        # Method 2: Regex patterns
        pattern_entities = self._extract_pattern_entities(text, document_id)
        entities.extend(pattern_entities)
        
        # Method 3: Noun phrase extraction
        noun_entities = self._extract_noun_phrase_entities(text, document_id)
        entities.extend(noun_entities)
        
        # Deduplicate and merge similar entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_spacy_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        try:
            doc = nlp(text)
            
            for ent in doc.ents:
                entity_id = self._generate_entity_id(ent.text, ent.label_)
                
                entity = Entity(
                    entity_id=entity_id,
                    name=ent.text.strip(),
                    entity_type=ent.label_,
                    aliases={ent.text.strip()},
                    attributes={
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                        'spacy_label': ent.label_
                    },
                    mentions=[{
                        'document_id': document_id,
                        'text': ent.text,
                        'context': text[max(0, ent.start_char-50):ent.end_char+50],
                        'start_char': ent.start_char,
                        'end_char': ent.end_char
                    }],
                    confidence=0.8  # spaCy entities generally high confidence
                )
                
                # Generate embedding if available
                if self.sentence_model:
                    entity.embedding = self.sentence_model.encode([ent.text])[0]
                
                entities.append(entity)
        
        except Exception as e:
            logging.warning(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    def _extract_pattern_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group().strip()
                    entity_id = self._generate_entity_id(entity_text, entity_type)
                    
                    entity = Entity(
                        entity_id=entity_id,
                        name=entity_text,
                        entity_type=entity_type,
                        aliases={entity_text},
                        attributes={
                            'pattern_matched': pattern,
                            'start_char': match.start(),
                            'end_char': match.end()
                        },
                        mentions=[{
                            'document_id': document_id,
                            'text': entity_text,
                            'context': text[max(0, match.start()-50):match.end()+50],
                            'start_char': match.start(),
                            'end_char': match.end()
                        }],
                        confidence=0.9  # Pattern matches are high confidence
                    )
                    
                    # Generate embedding if available
                    if self.sentence_model:
                        entity.embedding = self.sentence_model.encode([entity_text])[0]
                    
                    entities.append(entity)
        
        return entities
    
    def _extract_noun_phrase_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract potential entities from noun phrases."""
        entities = []
        
        if not SPACY_AVAILABLE:
            return entities
        
        try:
            doc = nlp(text)
            
            # Extract noun chunks that might be entities
            for chunk in doc.noun_chunks:
                # Filter out common words and short phrases
                if (len(chunk.text.strip()) > 2 and 
                    not chunk.text.lower().strip() in {'the', 'a', 'an', 'this', 'that', 'these', 'those'} and
                    not chunk.root.pos_ in {'PRON', 'DET'}):
                    
                    entity_text = chunk.text.strip()
                    entity_id = self._generate_entity_id(entity_text, "CONCEPT")
                    
                    entity = Entity(
                        entity_id=entity_id,
                        name=entity_text,
                        entity_type="CONCEPT",
                        aliases={entity_text},
                        attributes={
                            'pos_tags': [token.pos_ for token in chunk],
                            'start_char': chunk.start_char,
                            'end_char': chunk.end_char,
                            'extraction_method': 'noun_phrase'
                        },
                        mentions=[{
                            'document_id': document_id,
                            'text': entity_text,
                            'context': text[max(0, chunk.start_char-50):chunk.end_char+50],
                            'start_char': chunk.start_char,
                            'end_char': chunk.end_char
                        }],
                        confidence=0.6  # Lower confidence for noun phrases
                    )
                    
                    # Generate embedding if available
                    if self.sentence_model:
                        entity.embedding = self.sentence_model.encode([entity_text])[0]
                    
                    entities.append(entity)
        
        except Exception as e:
            logging.warning(f"Noun phrase extraction failed: {e}")
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate and merge similar entities."""
        if not entities:
            return entities
        
        # Group entities by normalized name
        entity_groups = defaultdict(list)
        
        for entity in entities:
            normalized_name = entity.name.lower().strip()
            entity_groups[normalized_name].append(entity)
        
        deduplicated = []
        
        for group_entities in entity_groups.values():
            if len(group_entities) == 1:
                deduplicated.append(group_entities[0])
            else:
                # Merge entities with same normalized name
                merged_entity = self._merge_entities(group_entities)
                deduplicated.append(merged_entity)
        
        return deduplicated
    
    def _merge_entities(self, entities: List[Entity]) -> Entity:
        """Merge multiple entities into one."""
        # Use the entity with highest confidence as base
        base_entity = max(entities, key=lambda e: e.confidence)
        
        # Merge aliases
        all_aliases = set()
        all_mentions = []
        all_attributes = {}
        
        for entity in entities:
            all_aliases.update(entity.aliases)
            all_mentions.extend(entity.mentions)
            all_attributes.update(entity.attributes)
        
        # Prefer more specific entity types
        entity_type_priority = {
            'PERSON': 5, 'ORG': 4, 'GPE': 3, 'LOCATION': 3, 
            'EVENT': 2, 'CONCEPT': 1, 'EMAIL': 4, 'PHONE': 4
        }
        
        best_type = max(entities, key=lambda e: entity_type_priority.get(e.entity_type, 0)).entity_type
        
        merged_entity = Entity(
            entity_id=base_entity.entity_id,
            name=base_entity.name,
            entity_type=best_type,
            aliases=all_aliases,
            attributes=all_attributes,
            mentions=all_mentions,
            confidence=max(e.confidence for e in entities),
            embedding=base_entity.embedding
        )
        
        return merged_entity
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a unique entity ID."""
        text = f"{name.lower().strip()}_{entity_type}"
        return hashlib.md5(text.encode()).hexdigest()[:12]


class RelationExtractor:
    """Advanced relation extraction between entities."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.relation_patterns = self._load_relation_patterns()
    
    def _load_relation_patterns(self) -> Dict[str, List[str]]:
        """Load relation extraction patterns."""
        return {
            'WORKS_FOR': [
                r'{entity1}.*(?:works? for|employed by|employee of).*{entity2}',
                r'{entity2}.*(?:employs?|hires?).*{entity1}'
            ],
            'LOCATED_IN': [
                r'{entity1}.*(?:located in|based in|situated in).*{entity2}',
                r'{entity1}.*(?:in|at).*{entity2}'
            ],
            'FOUNDED_BY': [
                r'{entity1}.*(?:founded by|established by|created by).*{entity2}',
                r'{entity2}.*(?:founded|established|created).*{entity1}'
            ],
            'PART_OF': [
                r'{entity1}.*(?:part of|division of|subsidiary of).*{entity2}',
                r'{entity2}.*(?:includes?|contains?).*{entity1}'
            ],
            'RELATED_TO': [
                r'{entity1}.*(?:related to|associated with|connected to).*{entity2}',
                r'{entity1}.*(?:and|with).*{entity2}'
            ]
        }
    
    def extract_relations(self, text: str, entities: List[Entity], document_id: str = None) -> List[Relation]:
        """Extract relations between entities in text."""
        relations = []
        
        # Method 1: Pattern-based extraction
        pattern_relations = self._extract_pattern_relations(text, entities, document_id)
        relations.extend(pattern_relations)
        
        # Method 2: Dependency parsing (if spaCy available)
        if SPACY_AVAILABLE:
            dependency_relations = self._extract_dependency_relations(text, entities, document_id)
            relations.extend(dependency_relations)
        
        # Method 3: Co-occurrence based relations
        cooccurrence_relations = self._extract_cooccurrence_relations(text, entities, document_id)
        relations.extend(cooccurrence_relations)
        
        # Deduplicate relations
        relations = self._deduplicate_relations(relations)
        
        return relations
    
    def _extract_pattern_relations(self, text: str, entities: List[Entity], document_id: str) -> List[Relation]:
        """Extract relations using predefined patterns."""
        relations = []
        
        # Create entity lookup
        entity_lookup = {entity.name.lower(): entity for entity in entities}
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                # Try all entity pairs
                for i, entity1 in enumerate(entities):
                    for entity2 in entities[i+1:]:
                        # Create pattern with entity placeholders
                        pattern_text = pattern.replace('{entity1}', re.escape(entity1.name))
                        pattern_text = pattern_text.replace('{entity2}', re.escape(entity2.name))
                        
                        matches = re.finditer(pattern_text, text, re.IGNORECASE | re.DOTALL)
                        
                        for match in matches:
                            relation_id = self._generate_relation_id(entity1.entity_id, entity2.entity_id, relation_type)
                            
                            relation = Relation(
                                relation_id=relation_id,
                                source_entity=entity1.entity_id,
                                target_entity=entity2.entity_id,
                                relation_type=relation_type,
                                confidence=0.8,
                                evidence=[match.group()],
                                attributes={
                                    'extraction_method': 'pattern',
                                    'pattern_used': pattern,
                                    'document_id': document_id
                                }
                            )
                            
                            relations.append(relation)
        
        return relations
    
    def _extract_dependency_relations(self, text: str, entities: List[Entity], document_id: str) -> List[Relation]:
        """Extract relations using dependency parsing."""
        relations = []
        
        try:
            doc = nlp(text)
            
            # Create entity spans lookup
            entity_spans = {}
            for entity in entities:
                for mention in entity.mentions:
                    if mention.get('document_id') == document_id:
                        start_char = mention.get('start_char', 0)
                        end_char = mention.get('end_char', 0)
                        entity_spans[(start_char, end_char)] = entity
            
            # Analyze dependency relations
            for token in doc:
                if token.dep_ in {'nsubj', 'dobj', 'pobj', 'compound'}:
                    # Find entities related through dependencies
                    head_entities = self._find_entities_in_span(token.head, entity_spans)
                    child_entities = self._find_entities_in_span(token, entity_spans)
                    
                    for head_entity in head_entities:
                        for child_entity in child_entities:
                            if head_entity.entity_id != child_entity.entity_id:
                                relation_type = self._infer_relation_type(token.dep_, token.head.pos_, token.pos_)
                                
                                if relation_type:
                                    relation_id = self._generate_relation_id(head_entity.entity_id, child_entity.entity_id, relation_type)
                                    
                                    relation = Relation(
                                        relation_id=relation_id,
                                        source_entity=head_entity.entity_id,
                                        target_entity=child_entity.entity_id,
                                        relation_type=relation_type,
                                        confidence=0.6,
                                        evidence=[token.sent.text],
                                        attributes={
                                            'extraction_method': 'dependency',
                                            'dependency_relation': token.dep_,
                                            'document_id': document_id
                                        }
                                    )
                                    
                                    relations.append(relation)
        
        except Exception as e:
            logging.warning(f"Dependency relation extraction failed: {e}")
        
        return relations
    
    def _extract_cooccurrence_relations(self, text: str, entities: List[Entity], document_id: str) -> List[Relation]:
        """Extract relations based on entity co-occurrence."""
        relations = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Find entities mentioned in this sentence
            mentioned_entities = []
            for entity in entities:
                if any(alias.lower() in sentence.lower() for alias in entity.aliases):
                    mentioned_entities.append(entity)
            
            # Create co-occurrence relations for entities in same sentence
            if len(mentioned_entities) >= 2:
                for i, entity1 in enumerate(mentioned_entities):
                    for entity2 in mentioned_entities[i+1:]:
                        relation_id = self._generate_relation_id(entity1.entity_id, entity2.entity_id, "CO_OCCURS")
                        
                        relation = Relation(
                            relation_id=relation_id,
                            source_entity=entity1.entity_id,
                            target_entity=entity2.entity_id,
                            relation_type="CO_OCCURS",
                            confidence=0.4,  # Lower confidence for co-occurrence
                            evidence=[sentence],
                            attributes={
                                'extraction_method': 'cooccurrence',
                                'sentence_context': sentence,
                                'document_id': document_id
                            }
                        )
                        
                        relations.append(relation)
        
        return relations
    
    def _find_entities_in_span(self, token, entity_spans: Dict) -> List[Entity]:
        """Find entities that overlap with a token span."""
        entities = []
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        for (start, end), entity in entity_spans.items():
            if start <= token_start < end or start < token_end <= end:
                entities.append(entity)
        
        return entities
    
    def _infer_relation_type(self, dep: str, head_pos: str, child_pos: str) -> Optional[str]:
        """Infer relation type from dependency information."""
        dep_mapping = {
            'nsubj': 'SUBJECT_OF',
            'dobj': 'OBJECT_OF', 
            'pobj': 'RELATED_TO',
            'compound': 'PART_OF'
        }
        
        return dep_mapping.get(dep)
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations."""
        seen_relations = set()
        deduplicated = []
        
        for relation in relations:
            # Create a key for deduplication
            key = (relation.source_entity, relation.target_entity, relation.relation_type)
            reverse_key = (relation.target_entity, relation.source_entity, relation.relation_type)
            
            if key not in seen_relations and reverse_key not in seen_relations:
                seen_relations.add(key)
                deduplicated.append(relation)
            else:
                # Merge evidence if relation already exists
                for existing_relation in deduplicated:
                    if ((existing_relation.source_entity == relation.source_entity and 
                         existing_relation.target_entity == relation.target_entity) or
                        (existing_relation.source_entity == relation.target_entity and 
                         existing_relation.target_entity == relation.source_entity)):
                        existing_relation.evidence.extend(relation.evidence)
                        existing_relation.confidence = max(existing_relation.confidence, relation.confidence)
                        break
        
        return deduplicated
    
    def _generate_relation_id(self, entity1_id: str, entity2_id: str, relation_type: str) -> str:
        """Generate a unique relation ID."""
        # Sort entity IDs to ensure consistent ID regardless of order
        sorted_ids = tuple(sorted([entity1_id, entity2_id]))
        text = f"{sorted_ids[0]}_{sorted_ids[1]}_{relation_type}"
        return hashlib.md5(text.encode()).hexdigest()[:12]


class KnowledgeGraph:
    """Knowledge graph storage and operations."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        
    def add_entity(self, entity: Entity):
        """Add an entity to the knowledge graph."""
        self.entities[entity.entity_id] = entity
        
        # Add to graph
        self.graph.add_node(entity.entity_id, 
                           name=entity.name,
                           type=entity.entity_type,
                           confidence=entity.confidence)
        
        # Update indexes
        self.entity_index[entity.name.lower()].add(entity.entity_id)
        for alias in entity.aliases:
            self.entity_index[alias.lower()].add(entity.entity_id)
    
    def add_relation(self, relation: Relation):
        """Add a relation to the knowledge graph."""
        self.relations[relation.relation_id] = relation
        
        # Add edge to graph
        self.graph.add_edge(relation.source_entity, 
                           relation.target_entity,
                           relation_type=relation.relation_type,
                           confidence=relation.confidence,
                           relation_id=relation.relation_id)
    
    def find_entities(self, name: str, fuzzy: bool = True) -> List[Entity]:
        """Find entities by name."""
        entities = []
        name_lower = name.lower()
        
        # Exact match
        if name_lower in self.entity_index:
            for entity_id in self.entity_index[name_lower]:
                entities.append(self.entities[entity_id])
        
        # Fuzzy match if no exact match
        if not entities and fuzzy:
            for indexed_name, entity_ids in self.entity_index.items():
                if name_lower in indexed_name or indexed_name in name_lower:
                    for entity_id in entity_ids:
                        entities.append(self.entities[entity_id])
        
        return entities
    
    def find_related_entities(self, entity_id: str, max_hops: int = 2, min_confidence: float = 0.3) -> List[Tuple[Entity, List[str]]]:
        """Find entities related to the given entity within max_hops."""
        if entity_id not in self.graph:
            return []
        
        related = []
        
        try:
            # Use NetworkX to find paths
            for target_id in self.graph.nodes():
                if target_id == entity_id:
                    continue
                
                try:
                    # Find shortest path
                    if nx.has_path(self.graph, entity_id, target_id):
                        path = nx.shortest_path(self.graph, entity_id, target_id)
                        
                        if len(path) <= max_hops + 1:  # +1 because path includes source
                            # Check if path has minimum confidence
                            path_confidence = self._calculate_path_confidence(path)
                            
                            if path_confidence >= min_confidence:
                                related.append((self.entities[target_id], path[1:]))  # Exclude source
                
                except nx.NetworkXNoPath:
                    continue
        
        except Exception as e:
            logging.warning(f"Related entity search failed: {e}")
        
        # Sort by confidence and proximity
        related.sort(key=lambda x: (len(x[1]), -x[0].confidence))
        
        return related
    
    def _calculate_path_confidence(self, path: List[str]) -> float:
        """Calculate confidence score for a path between entities."""
        if len(path) < 2:
            return 0.0
        
        confidences = []
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # Get edge data
            if self.graph.has_edge(source, target):
                edge_data = self.graph[source][target]
                if edge_data:
                    # Get the first edge (in case of multiple edges)
                    first_edge = next(iter(edge_data.values()))
                    confidences.append(first_edge.get('confidence', 0.5))
        
        # Return average confidence
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Query the knowledge graph using natural language."""
        results = []
        
        # Extract entities mentioned in query
        query_entities = []
        for name, entity_ids in self.entity_index.items():
            if name in query.lower():
                for entity_id in entity_ids:
                    query_entities.append(self.entities[entity_id])
        
        # For each entity, find related information
        for entity in query_entities:
            related_entities = self.find_related_entities(entity.entity_id)
            
            result = {
                'entity': entity.to_dict(),
                'related_entities': []
            }
            
            for related_entity, path in related_entities[:10]:  # Limit results
                result['related_entities'].append({
                    'entity': related_entity.to_dict(),
                    'connection_path': path,
                    'relationship_strength': len(path)
                })
            
            results.append(result)
        
        return results
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            stats = {
                'num_entities': len(self.entities),
                'num_relations': len(self.relations),
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'entity_types': Counter(e.entity_type for e in self.entities.values()),
                'relation_types': Counter(r.relation_type for r in self.relations.values()),
                'average_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
                'connected_components': nx.number_weakly_connected_components(self.graph),
                'graph_density': nx.density(self.graph)
            }
            
            # Most connected entities
            degree_centrality = nx.degree_centrality(self.graph)
            most_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            stats['most_connected_entities'] = [
                {
                    'entity_id': entity_id,
                    'name': self.entities[entity_id].name if entity_id in self.entities else 'Unknown',
                    'centrality': centrality
                }
                for entity_id, centrality in most_connected
            ]
            
            return stats
            
        except Exception as e:
            logging.warning(f"Graph statistics calculation failed: {e}")
            return {'error': str(e)}
    
    def export_graph(self, filepath: str, format: str = 'json') -> bool:
        """Export the knowledge graph to a file."""
        try:
            if format.lower() == 'json':
                export_data = {
                    'entities': [entity.to_dict() for entity in self.entities.values()],
                    'relations': [relation.to_dict() for relation in self.relations.values()],
                    'metadata': {
                        'export_timestamp': datetime.now().isoformat(),
                        'statistics': self.get_graph_statistics()
                    }
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'gexf':
                # Export as GEXF for Gephi
                nx.write_gexf(self.graph, filepath)
            
            elif format.lower() == 'pickle':
                # Export as pickle for Python
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'entities': self.entities,
                        'relations': self.relations,
                        'graph': self.graph,
                        'entity_index': dict(self.entity_index)
                    }, f)
            
            return True
            
        except Exception as e:
            logging.error(f"Graph export failed: {e}")
            return False


class GraphEnhancedRAG:
    """Main graph-enhanced RAG system."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.entity_extractor = EntityExtractor(config)
        self.relation_extractor = RelationExtractor(config)
        self.knowledge_graph = KnowledgeGraph(config)
        
        # Processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'relations_extracted': 0,
            'processing_time': 0.0
        }
    
    def process_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Process documents to build the knowledge graph."""
        start_time = datetime.now()
        
        for doc_idx, document in enumerate(documents):
            try:
                document_id = document.metadata.get('source', f'doc_{doc_idx}')
                
                # Extract entities
                entities = self.entity_extractor.extract_entities(document.page_content, document_id)
                
                # Add entities to graph
                for entity in entities:
                    self.knowledge_graph.add_entity(entity)
                
                # Extract relations
                relations = self.relation_extractor.extract_relations(
                    document.page_content, entities, document_id
                )
                
                # Add relations to graph
                for relation in relations:
                    self.knowledge_graph.add_relation(relation)
                
                # Update statistics
                self.processing_stats['entities_extracted'] += len(entities)
                self.processing_stats['relations_extracted'] += len(relations)
                
            except Exception as e:
                logging.warning(f"Failed to process document {doc_idx}: {e}")
        
        # Update final statistics
        self.processing_stats['documents_processed'] = len(documents)
        self.processing_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return self.processing_stats
    
    def enhanced_query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Enhanced query using knowledge graph."""
        try:
            # Standard graph query
            graph_results = self.knowledge_graph.query_graph(query)
            
            # Extract entities from query
            query_entities = self.entity_extractor.extract_entities(query, "query")
            
            # Find semantic expansions
            expanded_results = []
            for entity in query_entities:
                related = self.knowledge_graph.find_related_entities(entity.entity_id, max_hops=2)
                expanded_results.extend(related)
            
            # Combine and rank results
            all_results = {
                'direct_matches': graph_results,
                'semantic_expansion': [
                    {
                        'entity': related_entity.to_dict(),
                        'connection_path': path,
                        'relevance_score': 1.0 / (len(path) + 1)
                    }
                    for related_entity, path in expanded_results[:top_k]
                ],
                'query_analysis': {
                    'entities_found': len(query_entities),
                    'query_entities': [e.to_dict() for e in query_entities]
                },
                'graph_statistics': self.knowledge_graph.get_graph_statistics()
            }
            
            return all_results
            
        except Exception as e:
            logging.error(f"Enhanced query failed: {e}")
            return {'error': str(e)}
    
    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get comprehensive context for an entity."""
        try:
            entities = self.knowledge_graph.find_entities(entity_name)
            
            if not entities:
                return {'error': f'Entity "{entity_name}" not found'}
            
            entity = entities[0]  # Take the first match
            
            # Get related entities
            related = self.knowledge_graph.find_related_entities(entity.entity_id)
            
            # Get direct relations
            direct_relations = []
            for relation in self.knowledge_graph.relations.values():
                if relation.source_entity == entity.entity_id or relation.target_entity == entity.entity_id:
                    direct_relations.append(relation.to_dict())
            
            context = {
                'entity': entity.to_dict(),
                'direct_relations': direct_relations,
                'related_entities': [
                    {
                        'entity': related_entity.to_dict(),
                        'connection': path,
                        'distance': len(path)
                    }
                    for related_entity, path in related[:20]
                ],
                'context_summary': self._generate_entity_summary(entity, related[:10])
            }
            
            return context
            
        except Exception as e:
            logging.error(f"Entity context retrieval failed: {e}")
            return {'error': str(e)}
    
    def _generate_entity_summary(self, entity: Entity, related_entities: List[Tuple[Entity, List[str]]]) -> str:
        """Generate a summary of entity context."""
        try:
            summary_parts = []
            
            # Basic entity info
            summary_parts.append(f"{entity.name} is a {entity.entity_type.lower()}.")
            
            # Mentions
            if entity.mentions:
                doc_count = len(set(mention.get('document_id', 'unknown') for mention in entity.mentions))
                summary_parts.append(f"It appears in {len(entity.mentions)} mentions across {doc_count} documents.")
            
            # Key relationships
            if related_entities:
                relation_types = set()
                for _, path in related_entities:
                    # Get relations from path
                    for i in range(len(path)):
                        if path[i] in self.knowledge_graph.relations:
                            relation = self.knowledge_graph.relations[path[i]]
                            relation_types.add(relation.relation_type)
                
                if relation_types:
                    summary_parts.append(f"Key relationships include: {', '.join(sorted(relation_types))}.")
            
            # Most connected entities
            if related_entities:
                top_related = [entity for entity, _ in related_entities[:3]]
                names = [e.name for e in top_related]
                summary_parts.append(f"Closely connected to: {', '.join(names)}.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logging.warning(f"Entity summary generation failed: {e}")
            return f"Summary for {entity.name} could not be generated."
    
    def clear_graph(self):
        """Clear the knowledge graph."""
        self.knowledge_graph = KnowledgeGraph(self.config)
        self.processing_stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'relations_extracted': 0,
            'processing_time': 0.0
        }
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get current graph processing capabilities."""
        return {
            'entity_extraction': True,
            'relation_extraction': True,
            'named_entity_recognition': SPACY_AVAILABLE,
            'dependency_parsing': SPACY_AVAILABLE,
            'semantic_embeddings': SENTENCE_TRANSFORMERS_AVAILABLE,
            'advanced_clustering': SKLEARN_AVAILABLE,
            'graph_analysis': True
        }