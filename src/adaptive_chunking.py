"""
Adaptive Chunking System - Pro-Level Enhancement
Implements intelligent chunking based on document structure, content type, and semantic coherence.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Some adaptive chunking features will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Semantic chunking will be limited.")

from .config import Config
from .advanced_pdf_processor import PDFLayoutElement


@dataclass
class ChunkMetadata:
    """Enhanced metadata for adaptive chunks."""
    chunk_id: str
    document_id: str
    chunk_type: str  # paragraph, section, list, table, code, etc.
    structure_level: int  # hierarchical level (h1=1, h2=2, etc.)
    semantic_coherence: float  # 0-1 score
    keyword_density: float
    readability_score: float
    content_complexity: str  # simple, moderate, complex
    parent_section: Optional[str] = None
    child_sections: List[str] = None
    cross_references: List[str] = None
    # PDF-specific metadata
    pdf_layout_info: Optional[Dict[str, Any]] = None
    column_position: Optional[int] = None
    is_multi_column: bool = False
    
    def __post_init__(self):
        if self.child_sections is None:
            self.child_sections = []
        if self.cross_references is None:
            self.cross_references = []


@dataclass
class AdaptiveChunk:
    """Represents an intelligently chunked piece of content."""
    content: str
    metadata: ChunkMetadata
    original_document: Document
    chunk_size: int
    overlap_size: int
    confidence_score: float
    related_chunks: List[str] = None
    
    def __post_init__(self):
        if self.related_chunks is None:
            self.related_chunks = []
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format."""
        return Document(
            page_content=self.content,
            metadata={
                **self.original_document.metadata,
                'chunk_id': self.metadata.chunk_id,
                'chunk_type': self.metadata.chunk_type,
                'structure_level': self.metadata.structure_level,
                'semantic_coherence': self.metadata.semantic_coherence,
                'content_complexity': self.metadata.content_complexity,
                'adaptive_chunk': True
            }
        )


class DocumentStructureAnalyzer:
    """Analyzes document structure to inform chunking strategy."""
    
    def __init__(self):
        self.structure_patterns = {
            'heading_h1': r'^#\s+(.+)$',
            'heading_h2': r'^##\s+(.+)$', 
            'heading_h3': r'^###\s+(.+)$',
            'heading_h4': r'^####\s+(.+)$',
            'heading_h5': r'^#####\s+(.+)$',
            'heading_h6': r'^######\s+(.+)$',
            'bullet_list': r'^\s*[-*+]\s+(.+)$',
            'numbered_list': r'^\s*\d+\.\s+(.+)$',
            'code_block': r'^```[\s\S]*?```$',
            'inline_code': r'`([^`]+)`',
            'table_row': r'^\|(.+)\|$',
            'blockquote': r'^>\s+(.+)$',
            'horizontal_rule': r'^---+$',
            'link': r'\[([^\]]+)\]\([^)]+\)',
            'bold': r'\*\*([^*]+)\*\*',
            'italic': r'\*([^*]+)\*',
            'section_break': r'^={3,}$|^-{3,}$'
        }
    
    def analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure and return structural information."""
        lines = content.split('\n')
        structure_analysis = {
            'total_lines': len(lines),
            'structure_elements': defaultdict(list),
            'hierarchy': [],
            'sections': [],
            'content_types': defaultdict(int),
            'complexity_indicators': {}
        }
        
        current_section = None
        section_stack = []
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Detect structure elements
            for element_type, pattern in self.structure_patterns.items():
                matches = re.findall(pattern, line, re.MULTILINE)
                if matches:
                    structure_analysis['structure_elements'][element_type].append({
                        'line': line_num,
                        'content': matches[0] if matches else line_stripped,
                        'full_line': line
                    })
                    structure_analysis['content_types'][element_type] += 1
                    
                    # Track hierarchy for headings
                    if element_type.startswith('heading_'):
                        level = int(element_type.split('_h')[1])
                        heading_info = {
                            'level': level,
                            'title': matches[0] if matches else line_stripped,
                            'line': line_num
                        }
                        
                        # Manage section stack
                        while section_stack and section_stack[-1]['level'] >= level:
                            section_stack.pop()
                        
                        section_stack.append(heading_info)
                        structure_analysis['hierarchy'].append(heading_info)
                        current_section = heading_info['title']
            
            # Track sections
            if current_section and line_stripped:
                if not structure_analysis['sections'] or structure_analysis['sections'][-1]['title'] != current_section:
                    structure_analysis['sections'].append({
                        'title': current_section,
                        'start_line': line_num,
                        'content_lines': [line_num]
                    })
                else:
                    structure_analysis['sections'][-1]['content_lines'].append(line_num)
        
        # Calculate complexity indicators
        structure_analysis['complexity_indicators'] = self._calculate_complexity_indicators(
            structure_analysis, content
        )
        
        return structure_analysis
    
    def _calculate_complexity_indicators(self, structure: Dict[str, Any], content: str) -> Dict[str, float]:
        """Calculate various complexity indicators for the document."""
        indicators = {}
        
        # Structural complexity
        total_elements = sum(structure['content_types'].values())
        indicators['structural_density'] = total_elements / max(structure['total_lines'], 1)
        
        # Hierarchy depth
        max_heading_level = 0
        for heading in structure['hierarchy']:
            max_heading_level = max(max_heading_level, heading['level'])
        indicators['hierarchy_depth'] = max_heading_level
        
        # Content type diversity
        unique_types = len(structure['content_types'])
        indicators['content_diversity'] = unique_types / len(self.structure_patterns)
        
        # Text characteristics
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        indicators['avg_sentence_length'] = len(words) / max(len(sentences), 1)
        indicators['vocabulary_richness'] = len(set(words)) / max(len(words), 1)
        
        # Technical content indicators
        technical_patterns = [
            r'\b(?:API|SDK|HTTP|JSON|XML|SQL|CPU|GPU|RAM|URL|URI)\b',
            r'\b(?:function|class|method|variable|parameter|argument)\b',
            r'\b(?:algorithm|implementation|optimization|performance)\b',
            r'[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)',  # Function calls
            r'\b\d+\.\d+\.\d+\b',  # Version numbers
        ]
        
        technical_matches = 0
        for pattern in technical_patterns:
            technical_matches += len(re.findall(pattern, content, re.IGNORECASE))
        
        indicators['technical_density'] = technical_matches / max(len(words), 1)
        
        return indicators


class SemanticCoherenceAnalyzer:
    """Analyzes semantic coherence to optimize chunk boundaries."""
    
    def __init__(self):
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logging.warning(f"Failed to load sentence transformer: {e}")
    
    def calculate_coherence_score(self, text_segments: List[str]) -> float:
        """Calculate semantic coherence score for a group of text segments."""
        if len(text_segments) < 2:
            return 1.0
        
        if self.sentence_model:
            return self._semantic_coherence(text_segments)
        else:
            return self._lexical_coherence(text_segments)
    
    def _semantic_coherence(self, segments: List[str]) -> float:
        """Calculate semantic coherence using sentence embeddings."""
        try:
            embeddings = self.sentence_model.encode(segments)
            similarities = []
            
            # Calculate pairwise similarities
            for i in range(len(embeddings) - 1):
                for j in range(i + 1, len(embeddings)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logging.warning(f"Semantic coherence calculation failed: {e}")
            return self._lexical_coherence(segments)
    
    def _lexical_coherence(self, segments: List[str]) -> float:
        """Calculate lexical coherence using TF-IDF similarity."""
        if not SKLEARN_AVAILABLE or len(segments) < 2:
            return 0.5  # Default moderate coherence
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(segments)
            
            similarities = []
            for i in range(tfidf_matrix.shape[0] - 1):
                for j in range(i + 1, tfidf_matrix.shape[0]):
                    sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.5
    
    def find_optimal_boundaries(self, sentences: List[str], max_chunk_size: int) -> List[int]:
        """Find optimal chunk boundaries based on semantic coherence."""
        if len(sentences) <= 1:
            return [0, len(sentences)]
        
        # Calculate coherence scores between adjacent sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            score = self.calculate_coherence_score([sentences[i], sentences[i + 1]])
            coherence_scores.append(score)
        
        # Find low-coherence points as potential boundaries
        mean_coherence = np.mean(coherence_scores)
        std_coherence = np.std(coherence_scores)
        threshold = mean_coherence - (0.5 * std_coherence)
        
        boundaries = [0]
        current_size = 0
        
        for i, score in enumerate(coherence_scores):
            current_size += len(sentences[i])
            
            # Add boundary if coherence is low or size limit reached
            if (score < threshold or current_size >= max_chunk_size) and i > boundaries[-1]:
                boundaries.append(i + 1)
                current_size = 0
        
        if boundaries[-1] != len(sentences):
            boundaries.append(len(sentences))
        
        return boundaries


class AdaptiveChunker:
    """Main adaptive chunking system that combines structure and semantic analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.coherence_analyzer = SemanticCoherenceAnalyzer()
        
        # Chunking strategies by content type
        self.chunking_strategies = {
            'narrative': {'base_size': 1000, 'overlap': 200, 'coherence_weight': 0.8},
            'technical': {'base_size': 800, 'overlap': 150, 'coherence_weight': 0.6},
            'structured': {'base_size': 600, 'overlap': 100, 'coherence_weight': 0.4},
            'mixed': {'base_size': 800, 'overlap': 150, 'coherence_weight': 0.7}
        }
    
    def chunk_documents(self, documents: List[Document]) -> List[AdaptiveChunk]:
        """Chunk documents using adaptive strategies."""
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            try:
                doc_chunks = self._chunk_single_document(document, doc_idx)
                all_chunks.extend(doc_chunks)
            except Exception as e:
                logging.error(f"Failed to chunk document {doc_idx}: {e}")
                # Fallback to basic chunking
                fallback_chunks = self._fallback_chunking(document, doc_idx)
                all_chunks.extend(fallback_chunks)
        
        # Post-process chunks to establish relationships
        self._establish_chunk_relationships(all_chunks)
        
        return all_chunks
    
    def _chunk_single_document(self, document: Document, doc_idx: int) -> List[AdaptiveChunk]:
        """Chunk a single document adaptively."""
        content = document.page_content
        
        # Analyze document structure
        structure = self.structure_analyzer.analyze_structure(content)
        
        # Determine content type and strategy
        content_type = self._classify_content_type(structure)
        strategy = self.chunking_strategies[content_type]
        
        # Apply structure-aware chunking
        if structure['hierarchy']:
            chunks = self._hierarchical_chunking(document, structure, strategy, doc_idx)
        elif structure['sections']:
            chunks = self._section_based_chunking(document, structure, strategy, doc_idx)
        else:
            chunks = self._semantic_chunking(document, strategy, doc_idx)
        
        # Enhance chunks with metadata
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = self._enhance_chunk_metadata(chunk, structure, content_type)
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def chunk_pdf_with_layout(self, documents: List[Document]) -> List[Document]:
        """Enhanced chunking for PDFs that preserves layout information."""
        try:
            pdf_chunks = []
            
            for doc in documents:
                # Check if document has PDF layout information
                if doc.metadata.get('extraction_method') == 'pdfplumber_layout':
                    layout_chunks = self._chunk_pdf_layout_aware(doc)
                    pdf_chunks.extend(layout_chunks)
                else:
                    # Fall back to regular adaptive chunking
                    regular_chunks = self.adaptive_chunk_documents([doc])
                    pdf_chunks.extend(regular_chunks)
            
            return pdf_chunks
            
        except Exception as e:
            logging.warning(f"PDF layout-aware chunking failed: {e}")
            # Fallback to regular chunking
            return self.adaptive_chunk_documents(documents)
    
    def _chunk_pdf_layout_aware(self, document: Document) -> List[Document]:
        """Chunk a PDF document while respecting its layout structure."""
        try:
            content = document.page_content
            metadata = document.metadata
            layout_info = metadata.get('layout_info', {})
            
            chunks = []
            
            # Handle multi-column layouts
            if layout_info.get('layout_type') == 'multi_column':
                chunks = self._chunk_multicolumn_pdf(document)
            else:
                # Single column - use enhanced text chunking
                chunks = self._chunk_single_column_pdf(document)
            
            # Enhance chunks with PDF-specific metadata
            for chunk in chunks:
                self._add_pdf_metadata(chunk, layout_info)
            
            return chunks
            
        except Exception as e:
            logging.warning(f"PDF layout chunking failed: {e}")
            return [document]  # Return original if chunking fails
    
    def _chunk_multicolumn_pdf(self, document: Document) -> List[Document]:
        """Handle multi-column PDF layouts."""
        try:
            content = document.page_content
            metadata = document.metadata
            layout_info = metadata.get('layout_info', {})
            
            # Split content considering column boundaries
            column_boundaries = layout_info.get('column_boundaries', [])
            
            if not column_boundaries:
                return self._chunk_single_column_pdf(document)
            
            # For now, use regular chunking but mark as multi-column
            # Future enhancement: implement proper column-aware splitting
            chunks = self._split_content_smart(content, metadata)
            
            # Mark chunks as multi-column
            for chunk in chunks:
                chunk.metadata['is_multi_column'] = True
                chunk.metadata['column_boundaries'] = column_boundaries
            
            return chunks
            
        except Exception as e:
            logging.warning(f"Multi-column PDF chunking failed: {e}")
            return [document]
    
    def _chunk_single_column_pdf(self, document: Document) -> List[Document]:
        """Handle single-column PDF layouts with enhanced structure detection."""
        try:
            content = document.page_content
            metadata = document.metadata
            
            # Use structure-aware splitting
            chunks = self._split_content_smart(content, metadata)
            
            return chunks
            
        except Exception as e:
            logging.warning(f"Single-column PDF chunking failed: {e}")
            return [document]
    
    def _split_content_smart(self, content: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Smart content splitting that respects document structure."""
        try:
            # Enhanced separators for PDFs
            pdf_separators = [
                "\n\n\n",  # Major section breaks
                "\n\n",    # Paragraph breaks
                "\n• ",    # Bullet points
                "\n- ",    # Dash points
                "\n",      # Line breaks
                ". ",      # Sentence breaks
                " "        # Word breaks
            ]
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=pdf_separators,
                length_function=len
            )
            
            # Split the content
            text_chunks = splitter.split_text(content)
            
            # Convert to Document objects with enhanced metadata
            documents = []
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'chunk_method': 'pdf_structure_aware',
                    'chunk_length': len(chunk_text)
                })
                
                # Analyze chunk content
                chunk_analysis = self._analyze_chunk_structure(chunk_text)
                chunk_metadata.update(chunk_analysis)
                
                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logging.warning(f"Smart content splitting failed: {e}")
            return [Document(page_content=content, metadata=base_metadata)]
    
    def _analyze_chunk_structure(self, chunk_text: str) -> Dict[str, Any]:
        """Analyze the structure of a chunk to determine its characteristics."""
        try:
            analysis = {
                'has_headers': False,
                'has_lists': False,
                'has_tables': False,
                'has_citations': False,
                'structure_type': 'paragraph'
            }
            
            # Detect headers (lines that are short and followed by content)
            lines = chunk_text.split('\n')
            for i, line in enumerate(lines[:-1]):  # Exclude last line
                if len(line.strip()) < 50 and len(line.strip()) > 5:  # Potential header
                    next_line = lines[i + 1].strip()
                    if len(next_line) > len(line.strip()):
                        analysis['has_headers'] = True
                        analysis['structure_type'] = 'section'
                        break
            
            # Detect lists
            list_patterns = [r'^\s*[•\-\*]\s+', r'^\s*\d+\.\s+', r'^\s*[a-zA-Z]\.\s+']
            for pattern in list_patterns:
                if re.search(pattern, chunk_text, re.MULTILINE):
                    analysis['has_lists'] = True
                    if analysis['structure_type'] == 'paragraph':
                        analysis['structure_type'] = 'list'
                    break
            
            # Detect table-like content
            if '|' in chunk_text or '\t' in chunk_text:
                # Check for multiple aligned separators
                lines_with_separators = [line for line in lines if '|' in line or '\t' in line]
                if len(lines_with_separators) >= 2:
                    analysis['has_tables'] = True
                    analysis['structure_type'] = 'table'
            
            # Detect citations/references
            citation_patterns = [r'\[\d+\]', r'\(\w+\s+et\s+al\.,?\s+\d{4}\)', r'doi:', r'http://|https://']
            for pattern in citation_patterns:
                if re.search(pattern, chunk_text, re.IGNORECASE):
                    analysis['has_citations'] = True
                    break
            
            return analysis
            
        except Exception as e:
            logging.warning(f"Chunk structure analysis failed: {e}")
            return {'structure_type': 'paragraph'}
    
    def _add_pdf_metadata(self, chunk: Document, layout_info: Dict[str, Any]):
        """Add PDF-specific metadata to chunks."""
        try:
            chunk.metadata.update({
                'pdf_layout_info': layout_info,
                'is_pdf_chunk': True,
                'layout_type': layout_info.get('layout_type', 'single_column'),
                'num_columns': layout_info.get('columns', 1)
            })
            
            # Add column information if available
            if layout_info.get('layout_type') == 'multi_column':
                chunk.metadata['is_multi_column'] = True
                chunk.metadata['column_boundaries'] = layout_info.get('column_boundaries', [])
            
        except Exception as e:
            logging.warning(f"Adding PDF metadata failed: {e}")
    
    def get_pdf_chunking_summary(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get a summary of PDF chunking results."""
        try:
            pdf_chunks = [c for c in chunks if c.metadata.get('is_pdf_chunk', False)]
            
            if not pdf_chunks:
                return {"message": "No PDF chunks found"}
            
            summary = {
                "total_pdf_chunks": len(pdf_chunks),
                "layout_types": {},
                "structure_types": {},
                "multi_column_chunks": 0,
                "avg_chunk_length": 0,
                "has_tables": 0,
                "has_lists": 0,
                "has_headers": 0
            }
            
            total_length = 0
            
            for chunk in pdf_chunks:
                metadata = chunk.metadata
                
                # Count layout types
                layout_type = metadata.get('layout_type', 'unknown')
                summary['layout_types'][layout_type] = summary['layout_types'].get(layout_type, 0) + 1
                
                # Count structure types
                structure_type = metadata.get('structure_type', 'unknown')
                summary['structure_types'][structure_type] = summary['structure_types'].get(structure_type, 0) + 1
                
                # Count features
                if metadata.get('is_multi_column', False):
                    summary['multi_column_chunks'] += 1
                if metadata.get('has_tables', False):
                    summary['has_tables'] += 1
                if metadata.get('has_lists', False):
                    summary['has_lists'] += 1
                if metadata.get('has_headers', False):
                    summary['has_headers'] += 1
                
                total_length += len(chunk.page_content)
            
            summary['avg_chunk_length'] = total_length / len(pdf_chunks)
            
            return summary
            
        except Exception as e:
            logging.warning(f"PDF chunking summary failed: {e}")
            return {"error": str(e)}
    
    def _classify_content_type(self, structure: Dict[str, Any]) -> str:
        """Classify content type to determine chunking strategy."""
        content_types = structure['content_types']
        complexity = structure['complexity_indicators']
        
        # Technical content indicators
        technical_score = (
            content_types.get('code_block', 0) * 3 +
            content_types.get('inline_code', 0) * 1 +
            complexity.get('technical_density', 0) * 10
        )
        
        # Structured content indicators
        structured_score = (
            content_types.get('heading_h1', 0) * 2 +
            content_types.get('heading_h2', 0) * 1.5 +
            content_types.get('bullet_list', 0) * 1 +
            content_types.get('numbered_list', 0) * 1 +
            content_types.get('table_row', 0) * 2
        )
        
        # Narrative content indicators
        narrative_score = (
            complexity.get('avg_sentence_length', 0) / 20 +
            (1 - complexity.get('structural_density', 0)) * 5
        )
        
        scores = {
            'technical': technical_score,
            'structured': structured_score,
            'narrative': narrative_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return 'mixed'
        
        # Return type with highest score, or mixed if close
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores[0][1] - sorted_scores[1][1] < max_score * 0.3:
            return 'mixed'
        
        return sorted_scores[0][0]
    
    def _hierarchical_chunking(self, document: Document, structure: Dict[str, Any], 
                             strategy: Dict[str, Any], doc_idx: int) -> List[AdaptiveChunk]:
        """Chunk document based on hierarchical structure."""
        chunks = []
        content_lines = document.page_content.split('\n')
        
        for i, section in enumerate(structure['sections']):
            section_content = '\n'.join([
                content_lines[line] for line in section['content_lines'] 
                if line < len(content_lines)
            ])
            
            if len(section_content.strip()) == 0:
                continue
            
            # Determine chunk parameters based on section
            chunk_size = self._calculate_adaptive_chunk_size(
                section_content, strategy['base_size']
            )
            
            # Create chunk metadata
            metadata = ChunkMetadata(
                chunk_id=f"doc_{doc_idx}_section_{i}",
                document_id=f"doc_{doc_idx}",
                chunk_type="section",
                structure_level=self._get_section_level(section, structure['hierarchy']),
                semantic_coherence=0.8,  # High for structured content
                keyword_density=self._calculate_keyword_density(section_content),
                readability_score=self._calculate_readability(section_content),
                content_complexity=self._assess_complexity(section_content),
                parent_section=section.get('title', f"Section_{i}")
            )
            
            chunk = AdaptiveChunk(
                content=section_content,
                metadata=metadata,
                original_document=document,
                chunk_size=len(section_content),
                overlap_size=strategy['overlap'],
                confidence_score=0.9  # High confidence for structure-based chunks
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _section_based_chunking(self, document: Document, structure: Dict[str, Any],
                               strategy: Dict[str, Any], doc_idx: int) -> List[AdaptiveChunk]:
        """Chunk document based on sections without strict hierarchy."""
        chunks = []
        
        for i, section in enumerate(structure['sections']):
            content_lines = document.page_content.split('\n')
            section_content = '\n'.join([
                content_lines[line] for line in section['content_lines']
                if line < len(content_lines)
            ])
            
            if len(section_content.strip()) == 0:
                continue
            
            # Split large sections further if needed  
            if len(section_content) > strategy['base_size'] * 1.5:
                sub_chunks = self._split_large_section(
                    section_content, strategy, doc_idx, i, section['title']
                )
                chunks.extend(sub_chunks)
            else:
                metadata = ChunkMetadata(
                    chunk_id=f"doc_{doc_idx}_section_{i}",
                    document_id=f"doc_{doc_idx}",
                    chunk_type="section",
                    structure_level=1,
                    semantic_coherence=self.coherence_analyzer.calculate_coherence_score([section_content]),
                    keyword_density=self._calculate_keyword_density(section_content),
                    readability_score=self._calculate_readability(section_content),
                    content_complexity=self._assess_complexity(section_content),
                    parent_section=section.get('title', f"Section_{i}")
                )
                
                chunk = AdaptiveChunk(
                    content=section_content,
                    metadata=metadata,
                    original_document=document,
                    chunk_size=len(section_content),
                    overlap_size=strategy['overlap'],
                    confidence_score=0.8
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _semantic_chunking(self, document: Document, strategy: Dict[str, Any], 
                          doc_idx: int) -> List[AdaptiveChunk]:
        """Chunk document based on semantic coherence."""
        content = document.page_content
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Find optimal boundaries
        boundaries = self.coherence_analyzer.find_optimal_boundaries(
            sentences, strategy['base_size']
        )
        
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = '. '.join(chunk_sentences)
            
            if len(chunk_content.strip()) == 0:
                continue
            
            coherence_score = self.coherence_analyzer.calculate_coherence_score(chunk_sentences)
            
            metadata = ChunkMetadata(
                chunk_id=f"doc_{doc_idx}_semantic_{i}",
                document_id=f"doc_{doc_idx}",
                chunk_type="semantic",
                structure_level=0,
                semantic_coherence=coherence_score,
                keyword_density=self._calculate_keyword_density(chunk_content),
                readability_score=self._calculate_readability(chunk_content),
                content_complexity=self._assess_complexity(chunk_content)
            )
            
            chunk = AdaptiveChunk(
                content=chunk_content,
                metadata=metadata,
                original_document=document,
                chunk_size=len(chunk_content),
                overlap_size=strategy['overlap'],
                confidence_score=coherence_score
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_chunking(self, document: Document, doc_idx: int) -> List[AdaptiveChunk]:
        """Fallback to basic recursive chunking."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )
        
        basic_chunks = splitter.split_documents([document])
        adaptive_chunks = []
        
        for i, chunk_doc in enumerate(basic_chunks):
            metadata = ChunkMetadata(
                chunk_id=f"doc_{doc_idx}_fallback_{i}",
                document_id=f"doc_{doc_idx}",
                chunk_type="fallback",
                structure_level=0,
                semantic_coherence=0.5,
                keyword_density=self._calculate_keyword_density(chunk_doc.page_content),
                readability_score=self._calculate_readability(chunk_doc.page_content),
                content_complexity=self._assess_complexity(chunk_doc.page_content)
            )
            
            adaptive_chunk = AdaptiveChunk(
                content=chunk_doc.page_content,
                metadata=metadata,
                original_document=document,
                chunk_size=len(chunk_doc.page_content),
                overlap_size=150,
                confidence_score=0.4
            )
            
            adaptive_chunks.append(adaptive_chunk)
        
        return adaptive_chunks
    
    def _calculate_adaptive_chunk_size(self, content: str, base_size: int) -> int:
        """Calculate adaptive chunk size based on content characteristics."""
        # Adjust based on content density
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Adjust size based on sentence length
        if avg_sentence_length > 25:  # Long sentences
            return int(base_size * 1.2)
        elif avg_sentence_length < 10:  # Short sentences
            return int(base_size * 0.8)
        else:
            return base_size
    
    def _get_section_level(self, section: Dict[str, Any], hierarchy: List[Dict[str, Any]]) -> int:
        """Get the hierarchical level of a section."""
        section_title = section.get('title', '')
        
        for heading in hierarchy:
            if heading['title'] == section_title:
                return heading['level']
        
        return 1  # Default level
    
    def _split_large_section(self, content: str, strategy: Dict[str, Any], 
                           doc_idx: int, section_idx: int, section_title: str) -> List[AdaptiveChunk]:
        """Split large sections into smaller chunks."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunk_size = strategy['base_size']
        overlap = strategy['overlap']
        chunks = []
        
        current_chunk = ""
        chunk_idx = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Create chunk
                metadata = ChunkMetadata(
                    chunk_id=f"doc_{doc_idx}_section_{section_idx}_sub_{chunk_idx}",
                    document_id=f"doc_{doc_idx}",
                    chunk_type="subsection",
                    structure_level=2,
                    semantic_coherence=self.coherence_analyzer.calculate_coherence_score([current_chunk]),
                    keyword_density=self._calculate_keyword_density(current_chunk),
                    readability_score=self._calculate_readability(current_chunk),
                    content_complexity=self._assess_complexity(current_chunk),
                    parent_section=section_title
                )
                
                chunk = AdaptiveChunk(
                    content=current_chunk.strip(),
                    metadata=metadata,
                    original_document=None,  # Will be set later
                    chunk_size=len(current_chunk),
                    overlap_size=overlap,
                    confidence_score=0.7
                )
                
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-min(overlap // 5, len(words)):]  # Approximate overlap
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = sentence
                
                chunk_idx += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            metadata = ChunkMetadata(
                chunk_id=f"doc_{doc_idx}_section_{section_idx}_sub_{chunk_idx}",
                document_id=f"doc_{doc_idx}",
                chunk_type="subsection",
                structure_level=2,
                semantic_coherence=self.coherence_analyzer.calculate_coherence_score([current_chunk]),
                keyword_density=self._calculate_keyword_density(current_chunk),
                readability_score=self._calculate_readability(current_chunk),  
                content_complexity=self._assess_complexity(current_chunk),
                parent_section=section_title
            )
            
            chunk = AdaptiveChunk(
                content=current_chunk.strip(),
                metadata=metadata,
                original_document=None,
                chunk_size=len(current_chunk),
                overlap_size=overlap,
                confidence_score=0.7
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _establish_chunk_relationships(self, chunks: List[AdaptiveChunk]):
        """Establish relationships between chunks."""
        for i, chunk in enumerate(chunks):
            # Find related chunks based on semantic similarity and structure
            related_chunks = []
            
            for j, other_chunk in enumerate(chunks):
                if i != j and chunk.metadata.document_id == other_chunk.metadata.document_id:
                    # Same parent section
                    if (chunk.metadata.parent_section and 
                        chunk.metadata.parent_section == other_chunk.metadata.parent_section):
                        related_chunks.append(other_chunk.metadata.chunk_id)
                    
                    # Adjacent structure levels
                    elif abs(chunk.metadata.structure_level - other_chunk.metadata.structure_level) <= 1:
                        related_chunks.append(other_chunk.metadata.chunk_id)
            
            chunk.related_chunks = related_chunks[:5]  # Limit to top 5 related chunks
    
    def _calculate_keyword_density(self, content: str) -> float:
        """Calculate keyword density for content optimization."""
        words = content.lower().split()
        if not words:
            return 0.0
        
        # Simple keyword density based on word frequency
        word_freq = Counter(words)
        total_words = len(words)
        
        # Get top keywords (excluding common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_freq = {word: freq for word, freq in word_freq.items() 
                        if word not in stop_words and len(word) > 2}
        
        if not filtered_freq:
            return 0.0
        
        # Calculate density of most frequent meaningful word
        max_freq = max(filtered_freq.values())
        return max_freq / total_words
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate simple readability score."""
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        if not sentences or not words:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability based on sentence length
        # Shorter sentences = higher readability
        if avg_sentence_length <= 15:
            return 0.9
        elif avg_sentence_length <= 25:
            return 0.7
        elif avg_sentence_length <= 35:
            return 0.5
        else:
            return 0.3
    
    def _assess_complexity(self, content: str) -> str:
        """Assess content complexity level."""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        # Complexity indicators
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Technical terms
        technical_pattern = r'\b(?:implementation|optimization|algorithm|architecture|methodology|configuration|initialization|synchronization)\b'
        technical_count = len(re.findall(technical_pattern, content.lower()))
        technical_density = technical_count / max(len(words), 1)
        
        complexity_score = (
            (avg_word_length - 4) * 0.3 +
            (avg_sentence_length - 15) * 0.02 +
            technical_density * 10
        )
        
        if complexity_score < 0.5:
            return "simple"
        elif complexity_score < 1.5:
            return "moderate"
        else:
            return "complex"
    
    def get_chunking_analytics(self, chunks: List[AdaptiveChunk]) -> Dict[str, Any]:
        """Get analytics about the chunking process."""
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Basic statistics
        chunk_sizes = [chunk.chunk_size for chunk in chunks]
        coherence_scores = [chunk.metadata.semantic_coherence for chunk in chunks]
        confidence_scores = [chunk.confidence_score for chunk in chunks]
        
        # Type distribution
        type_distribution = Counter(chunk.metadata.chunk_type for chunk in chunks)
        complexity_distribution = Counter(chunk.metadata.content_complexity for chunk in chunks)
        
        # Structure levels
        structure_levels = [chunk.metadata.structure_level for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "chunk_size_stats": {
                "mean": np.mean(chunk_sizes),
                "median": np.median(chunk_sizes),
                "std": np.std(chunk_sizes),
                "min": np.min(chunk_sizes),
                "max": np.max(chunk_sizes)
            },
            "coherence_stats": {
                "mean": np.mean(coherence_scores),
                "median": np.median(coherence_scores),
                "std": np.std(coherence_scores)
            },
            "confidence_stats": {
                "mean": np.mean(confidence_scores),
                "median": np.median(confidence_scores),
                "std": np.std(confidence_scores)
            },
            "type_distribution": dict(type_distribution),
            "complexity_distribution": dict(complexity_distribution),
            "structure_depth": {
                "max_level": max(structure_levels) if structure_levels else 0,
                "avg_level": np.mean(structure_levels) if structure_levels else 0
            },
            "relationships": {
                "total_relationships": sum(len(chunk.related_chunks) for chunk in chunks),
                "avg_relationships_per_chunk": np.mean([len(chunk.related_chunks) for chunk in chunks])
            }
        }