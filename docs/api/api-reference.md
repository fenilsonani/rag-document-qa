# RAG System API Documentation

> **Complete API Reference for Multi-Modal Document Intelligence Platform**
> 
> **Version**: 2.0 Enhanced | **API Type**: Python Classes & Methods | **Status**: Production Ready

---

## 📋 Table of Contents

1. [API Overview](#api-overview)
2. [Core Classes](#core-classes)
3. [Document Processing APIs](#document-processing-apis)
4. [Multi-Modal Processing APIs](#multi-modal-processing-apis)
5. [Search & Query APIs](#search--query-apis)
6. [Configuration APIs](#configuration-apis)
7. [Response Formats](#response-formats)
8. [Error Handling](#error-handling)
9. [Usage Examples](#usage-examples)
10. [Best Practices](#best-practices)

---

## API Overview

### 🎯 API Architecture

The RAG System API provides a comprehensive interface for document processing, multi-modal analysis, and intelligent search capabilities.

#### API Design Principles
```yaml
Design Philosophy:
  ✅ Modular: Each component can be used independently
  ✅ Extensible: Easy to add new file formats and AI models
  ✅ Type-Safe: Full type hints and validation
  ✅ Async-Ready: Non-blocking operations where possible
  ✅ Error-Resilient: Comprehensive error handling and recovery

API Layers:
  - High-Level APIs: Simple interfaces for common tasks
  - Mid-Level APIs: Granular control over processing
  - Low-Level APIs: Direct access to individual components
  - Configuration APIs: System configuration and tuning
```

#### Quick Start Example
```python
from src.document_loader import DocumentProcessor
from src.enhanced_rag import EnhancedRAG

# Initialize the system
processor = DocumentProcessor()
rag_system = EnhancedRAG()

# Process a document
documents = processor.load_document("financial_report.pdf")

# Query the system
response = rag_system.query("What were the Q3 revenue figures?")
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.source_documents)} documents")
```

---

## Core Classes

### 📊 DocumentProcessor

**Primary class for document loading and multi-format processing**

#### Class Definition
```python
class DocumentProcessor:
    """
    Main class for document processing and format detection
    
    Handles 26+ file formats with advanced processing capabilities
    including PDF table/image extraction, Excel analysis, and AI-powered
    image processing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DocumentProcessor
        
        Args:
            config (Config, optional): Configuration object. If None, uses default config.
        """
```

#### Core Methods

##### `load_document(file_path: str) -> List[Document]`
```python
def load_document(self, file_path: str) -> List[Document]:
    """
    Load and process a single document with automatic format detection
    
    Args:
        file_path (str): Absolute path to the document file
        
    Returns:
        List[Document]: List of processed document chunks with metadata
        
    Raises:
        ValueError: If file type not supported or file not found
        RuntimeError: If processing fails
        
    Example:
        >>> processor = DocumentProcessor()
        >>> documents = processor.load_document("/path/to/report.pdf")
        >>> print(f"Loaded {len(documents)} document chunks")
    """
```

##### `get_multimodal_elements() -> List[MultiModalElement]`
```python
def get_multimodal_elements(self) -> List[MultiModalElement]:
    """
    Get all extracted multimodal elements (tables, images, charts)
    
    Returns:
        List[MultiModalElement]: All extracted elements with metadata
        
    Example:
        >>> elements = processor.get_multimodal_elements()
        >>> tables = [e for e in elements if e.element_type == "table"]
        >>> images = [e for e in elements if e.element_type == "image"]
        >>> print(f"Found {len(tables)} tables and {len(images)} images")
    """
```

##### `validate_file_support(file_path: str) -> Dict[str, Any]`
```python
def validate_file_support(self, file_path: str) -> Dict[str, Any]:
    """
    Check if file is supported and get processing information
    
    Args:
        file_path (str): Path to file for validation
        
    Returns:
        Dict[str, Any]: Validation result with processing details
        
    Example:
        >>> validation = processor.validate_file_support("document.xlsx")
        >>> print(f"Supported: {validation['is_supported']}")
        >>> print(f"Processing method: {validation['processing_method']}")
        >>> print(f"Features: {validation['available_features']}")
    """
```

##### `get_processing_summary() -> Dict[str, Any]`
```python
def get_processing_summary(self) -> Dict[str, Any]:
    """
    Get comprehensive processing summary and statistics
    
    Returns:
        Dict[str, Any]: Processing summary with metrics
        
    Example:
        >>> summary = processor.get_processing_summary()
        >>> print(f"Total elements: {summary['total_multimodal_elements']}")
        >>> print(f"Average confidence: {summary['avg_confidence']:.2f}")
    """
```

---

### 🤖 EnhancedRAG

**Advanced RAG system with multi-modal capabilities**

#### Class Definition
```python
class EnhancedRAG:
    """
    Enhanced RAG system supporting multi-modal document intelligence
    
    Provides advanced query processing across text, tables, images,
    and charts with intelligent result fusion and confidence scoring.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Enhanced RAG system
        
        Args:
            config (Config, optional): System configuration
        """
```

#### Core Methods

##### `query(question: str, **kwargs) -> RAGResponse`
```python
def query(
    self,
    question: str,
    include_multimodal: bool = True,
    max_results: int = 5,
    confidence_threshold: float = 0.7,
    modalities: List[str] = None
) -> RAGResponse:
    """
    Process a query with multi-modal search capabilities
    
    Args:
        question (str): User query
        include_multimodal (bool): Include multimodal content in search
        max_results (int): Maximum number of results to return
        confidence_threshold (float): Minimum confidence for results
        modalities (List[str]): Specific modalities to search ['text', 'tables', 'images']
        
    Returns:
        RAGResponse: Comprehensive response with answer and sources
        
    Example:
        >>> rag = EnhancedRAG()
        >>> response = rag.query(
        ...     "What are the Q3 revenue figures?",
        ...     include_multimodal=True,
        ...     max_results=3
        ... )
        >>> print(response.answer)
        >>> print(f"Confidence: {response.confidence_score}")
    """
```

##### `add_documents(documents: List[Document]) -> None`
```python
def add_documents(self, documents: List[Document]) -> None:
    """
    Add processed documents to the RAG system
    
    Args:
        documents (List[Document]): List of processed documents
        
    Example:
        >>> processor = DocumentProcessor()
        >>> documents = processor.load_document("report.pdf")
        >>> rag = EnhancedRAG()
        >>> rag.add_documents(documents)
    """
```

##### `search_multimodal_content(query: str) -> List[MultiModalElement]`
```python
def search_multimodal_content(
    self,
    query: str,
    element_types: List[str] = None,
    min_confidence: float = 0.6
) -> List[MultiModalElement]:
    """
    Search specifically within multimodal elements
    
    Args:
        query (str): Search query
        element_types (List[str]): Filter by element types ['table', 'image', 'chart']
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        List[MultiModalElement]: Matching multimodal elements
        
    Example:
        >>> elements = rag.search_multimodal_content(
        ...     "revenue chart",
        ...     element_types=["chart"],
        ...     min_confidence=0.8
        ... )
    """
```

---

### 🔧 UniversalFileProcessor

**Handles processing of non-PDF file formats**

#### Class Definition
```python
class UniversalFileProcessor:
    """
    Universal processor for 20+ non-PDF file formats
    
    Supports Excel, PowerPoint, images, JSON, XML, HTML, and more
    with intelligent format detection and specialized processing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Universal File Processor
        
        Args:
            config (Config, optional): Configuration object
        """
```

#### Core Methods

##### `process_file(file_path: str) -> FileProcessingResult`
```python
def process_file(self, file_path: str) -> FileProcessingResult:
    """
    Process any supported file format with automatic detection
    
    Args:
        file_path (str): Path to the file to process
        
    Returns:
        FileProcessingResult: Processing results with documents and multimodal elements
        
    Example:
        >>> processor = UniversalFileProcessor()
        >>> result = processor.process_file("data.xlsx")
        >>> if result.success:
        ...     print(f"Processed {len(result.documents)} sheets")
        ...     print(f"Extracted {len(result.multimodal_elements)} tables")
    """
```

##### `get_supported_formats() -> Dict[str, Dict[str, Any]]`
```python
def get_supported_formats(self) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about all supported file formats
    
    Returns:
        Dict[str, Dict[str, Any]]: Format categories with capabilities
        
    Example:
        >>> formats = processor.get_supported_formats()
        >>> for category, info in formats.items():
        ...     print(f"{category}: {info['extensions']}")
        ...     print(f"Features: {info['features']}")
    """
```

---

### 📊 AdvancedPDFProcessor

**Specialized PDF processing with multi-method extraction**

#### Class Definition
```python
class AdvancedPDFProcessor:
    """
    Professional-grade PDF processor with multi-method table and image extraction
    
    Uses pdfplumber, camelot-py, PyMuPDF, and tabula-py for maximum accuracy
    and comprehensive content extraction.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Advanced PDF Processor
        
        Args:
            config (Config): Configuration with PDF processing settings
        """
```

#### Core Methods

##### `process_pdf(file_path: str) -> Tuple[List[Document], List[MultiModalElement]]`
```python
def process_pdf(self, file_path: str) -> Tuple[List[Document], List[MultiModalElement]]:
    """
    Process PDF with advanced table and image extraction
    
    Args:
        file_path (str): Path to PDF file
        
    Returns:
        Tuple[List[Document], List[MultiModalElement]]: Documents and multimodal elements
        
    Example:
        >>> pdf_processor = AdvancedPDFProcessor(config)
        >>> documents, elements = pdf_processor.process_pdf("report.pdf")
        >>> tables = [e for e in elements if e.element_type == "table"]
        >>> images = [e for e in elements if e.element_type == "image"]
    """
```

##### `extract_tables(file_path: str, methods: List[str] = None) -> List[MultiModalElement]`
```python
def extract_tables(
    self,
    file_path: str,
    methods: List[str] = None,
    confidence_threshold: float = 0.7
) -> List[MultiModalElement]:
    """
    Extract tables using specified methods
    
    Args:
        file_path (str): Path to PDF file
        methods (List[str]): Extraction methods ['pdfplumber', 'camelot', 'pymupdf', 'tabula']
        confidence_threshold (float): Minimum confidence for inclusion
        
    Returns:
        List[MultiModalElement]: Extracted table elements
        
    Example:
        >>> tables = pdf_processor.extract_tables(
        ...     "financial_report.pdf",
        ...     methods=["camelot", "pdfplumber"],
        ...     confidence_threshold=0.8
        ... )
    """
```

---

## Document Processing APIs

### 📄 Document Loading and Processing

#### Batch Document Processing
```python
def load_multiple_documents(
    self,
    file_paths: List[str],
    parallel: bool = True,
    max_workers: int = 3
) -> List[Document]:
    """
    Load multiple documents with optional parallel processing
    
    Args:
        file_paths (List[str]): List of file paths to process
        parallel (bool): Enable parallel processing
        max_workers (int): Number of worker threads for parallel processing
        
    Returns:
        List[Document]: All processed documents
        
    Example:
        >>> files = ["report1.pdf", "data.xlsx", "presentation.pptx"]
        >>> documents = processor.load_multiple_documents(files, parallel=True)
    """
```

#### Document Validation
```python
def validate_files(
    self,
    file_paths: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Validate multiple files and get processing information
    
    Args:
        file_paths (List[str]): List of file paths to validate
        
    Returns:
        Dict[str, Dict[str, Any]]: Validation results for each file
        
    Example:
        >>> files = ["doc1.pdf", "doc2.xlsx", "invalid.xyz"]
        >>> validation_results = processor.validate_files(files)
        >>> for file, result in validation_results.items():
        ...     print(f"{file}: {'✅' if result['is_supported'] else '❌'}")
    """
```

#### Content Filtering and Search
```python
def filter_documents(
    self,
    documents: List[Document],
    filters: Dict[str, Any]
) -> List[Document]:
    """
    Filter documents based on metadata and content criteria
    
    Args:
        documents (List[Document]): Documents to filter
        filters (Dict[str, Any]): Filter criteria
        
    Returns:
        List[Document]: Filtered documents
        
    Example:
        >>> filters = {
        ...     "file_type": ".pdf",
        ...     "min_confidence": 0.8,
        ...     "contains_tables": True
        ... }
        >>> filtered_docs = processor.filter_documents(documents, filters)
    """
```

---

## Multi-Modal Processing APIs

### 🖼️ Image and Visual Content Processing

#### Image Analysis
```python
class MultiModalProcessor:
    """
    Advanced multi-modal content processing with AI models
    """
    
    def analyze_image(
        self,
        image_path: str,
        ai_models: List[str] = ["blip", "detr", "ocr"]
    ) -> ImageAnalysisResult:
        """
        Comprehensive AI-powered image analysis
        
        Args:
            image_path (str): Path to image file
            ai_models (List[str]): AI models to use for analysis
            
        Returns:
            ImageAnalysisResult: Complete analysis results
            
        Example:
            >>> processor = MultiModalProcessor()
            >>> result = processor.analyze_image("chart.png")
            >>> print(f"Caption: {result.caption}")
            >>> print(f"Objects: {result.detected_objects}")
            >>> print(f"OCR Text: {result.extracted_text}")
        """
```

#### Chart Data Extraction
```python
def extract_chart_data(
    self,
    image_path: str,
    chart_type: str = None
) -> ChartAnalysisResult:
    """
    Extract data and insights from charts and visualizations
    
    Args:
        image_path (str): Path to chart image
        chart_type (str): Expected chart type for optimized processing
        
    Returns:
        ChartAnalysisResult: Extracted data and analysis
        
    Example:
        >>> result = processor.extract_chart_data("revenue_chart.png", "bar_chart")
        >>> print(f"Data points: {result.data_points}")
        >>> print(f"Trends: {result.trend_analysis}")
    """
```

#### Table Analysis
```python
def analyze_table_structure(
    self,
    table_element: MultiModalElement
) -> TableAnalysisResult:
    """
    Analyze table structure and perform data analysis
    
    Args:
        table_element (MultiModalElement): Extracted table element
        
    Returns:
        TableAnalysisResult: Structural and statistical analysis
        
    Example:
        >>> tables = processor.get_tables()
        >>> for table in tables:
        ...     analysis = processor.analyze_table_structure(table)
        ...     print(f"Columns: {analysis.column_count}")
        ...     print(f"Data types: {analysis.data_types}")
    """
```

---

## Search & Query APIs

### 🔍 Advanced Search Capabilities

#### Hybrid Search
```python
class HybridSearchEngine:
    """
    Advanced search combining multiple search strategies
    """
    
    def search(
        self,
        query: str,
        search_types: List[str] = ["semantic", "keyword", "multimodal"],
        max_results: int = 10,
        filters: Dict[str, Any] = None
    ) -> SearchResults:
        """
        Perform hybrid search across all content types
        
        Args:
            query (str): Search query
            search_types (List[str]): Types of search to perform
            max_results (int): Maximum results to return
            filters (Dict[str, Any]): Search filters
            
        Returns:
            SearchResults: Ranked search results with metadata
            
        Example:
            >>> search_engine = HybridSearchEngine()
            >>> results = search_engine.search(
            ...     "Q3 financial performance",
            ...     search_types=["semantic", "multimodal"],
            ...     filters={"file_type": ".pdf", "confidence_min": 0.8}
            ... )
        """
```

#### Cross-Modal Search
```python
def cross_modal_search(
    self,
    query: str,
    modalities: List[str] = ["text", "tables", "images", "charts"],
    fusion_strategy: str = "hybrid"
) -> CrossModalResults:
    """
    Search across different content modalities with result fusion
    
    Args:
        query (str): Search query
        modalities (List[str]): Content modalities to search
        fusion_strategy (str): Strategy for combining results
        
    Returns:
        CrossModalResults: Unified results from all modalities
        
    Example:
        >>> results = search_engine.cross_modal_search(
        ...     "revenue growth trends",
        ...     modalities=["text", "charts", "tables"]
        ... )
        >>> for result in results.unified_results:
        ...     print(f"{result.modality}: {result.content[:100]}...")
    """
```

#### Semantic Search
```python
def semantic_search(
    self,
    query: str,
    embedding_model: str = None,
    similarity_threshold: float = 0.7
) -> List[SemanticResult]:
    """
    Perform semantic similarity search using embeddings
    
    Args:
        query (str): Query text
        embedding_model (str): Embedding model to use
        similarity_threshold (float): Minimum similarity score
        
    Returns:
        List[SemanticResult]: Semantically similar content
        
    Example:
        >>> results = search_engine.semantic_search(
        ...     "financial performance metrics",
        ...     similarity_threshold=0.8
        ... )
    """
```

---

## Configuration APIs

### ⚙️ System Configuration

#### Configuration Management
```python
class Config:
    """
    System configuration with validation and environment variable support
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration
        
        Args:
            config_file (str): Path to configuration file
        """
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration parameters
        
        Args:
            updates (Dict[str, Any]): Configuration updates
            
        Example:
            >>> config = Config()
            >>> config.update_config({
            ...     "PDF_EXTRACTION_METHODS": ["camelot", "pdfplumber"],
            ...     "IMAGE_AI_ANALYSIS": True,
            ...     "CONFIDENCE_THRESHOLD": 0.8
            ... })
        """
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate current configuration
        
        Returns:
            Dict[str, Any]: Validation results and warnings
            
        Example:
            >>> validation = config.validate_config()
            >>> if validation["valid"]:
            ...     print("Configuration is valid")
            >>> else:
            ...     print(f"Errors: {validation['errors']}")
        """
```

#### Performance Configuration
```python
class PerformanceConfig:
    """
    Performance-specific configuration settings
    """
    
    def set_performance_mode(self, mode: str) -> None:
        """
        Set predefined performance mode
        
        Args:
            mode (str): Performance mode ['fast', 'balanced', 'accuracy']
            
        Example:
            >>> perf_config = PerformanceConfig()
            >>> perf_config.set_performance_mode("accuracy")  # Maximum accuracy
        """
    
    def optimize_for_hardware(self, hardware_info: Dict[str, Any]) -> None:
        """
        Automatically optimize configuration for hardware
        
        Args:
            hardware_info (Dict[str, Any]): System hardware information
            
        Example:
            >>> hardware = {"ram_gb": 16, "gpu": True, "cpu_cores": 8}
            >>> perf_config.optimize_for_hardware(hardware)
        """
```

---

## Response Formats

### 📋 Standard Response Objects

#### RAGResponse
```python
@dataclass
class RAGResponse:
    """
    Standard response from RAG query
    """
    answer: str                              # Generated answer
    source_documents: List[Document]         # Source documents used
    multimodal_elements: List[MultiModalElement]  # Related multimodal content
    confidence_score: float                  # Overall confidence (0.0-1.0)
    processing_time: float                   # Query processing time in seconds
    metadata: Dict[str, Any]                 # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        
    def to_json(self) -> str:
        """Convert to JSON string"""
```

#### FileProcessingResult
```python
@dataclass
class FileProcessingResult:
    """
    Result from file processing operations
    """
    success: bool                            # Processing success status
    documents: List[Document]                # Extracted document chunks
    multimodal_elements: List[MultiModalElement]  # Extracted multimodal content
    processing_time: float                   # Processing time in seconds
    error: Optional[str]                     # Error message if failed
    metadata: Dict[str, Any]                 # Processing metadata
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        return {
            "documents_created": len(self.documents),
            "tables_extracted": len([e for e in self.multimodal_elements if e.element_type == "table"]),
            "images_processed": len([e for e in self.multimodal_elements if e.element_type == "image"]),
            "processing_time": self.processing_time,
            "success_rate": 1.0 if self.success else 0.0
        }
```

#### MultiModalElement
```python
@dataclass
class MultiModalElement:
    """
    Represents extracted multimodal content
    """
    element_id: str                          # Unique identifier
    element_type: str                        # Type: 'table', 'image', 'chart'
    source_file: str                         # Source file path
    page_number: int                         # Page number (if applicable)
    text_description: str                    # Human-readable description
    raw_data: Any                           # Raw extracted data
    confidence_score: float                  # Extraction confidence (0.0-1.0)
    processing_method: str                   # Method used for extraction
    metadata: Dict[str, Any]                # Additional metadata
    
    def get_searchable_text(self) -> str:
        """Get text content for search indexing"""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
```

### 📊 Specialized Response Objects

#### ImageAnalysisResult
```python
@dataclass
class ImageAnalysisResult:
    """
    Result from AI-powered image analysis
    """
    image_path: str                          # Path to analyzed image
    caption: str                            # AI-generated caption
    detected_objects: List[Dict[str, Any]]   # DETR object detections
    extracted_text: str                      # OCR extracted text
    confidence_scores: Dict[str, float]      # Per-model confidence scores
    processing_metadata: Dict[str, Any]      # Processing details
    
    @property
    def overall_confidence(self) -> float:
        """Calculate overall analysis confidence"""
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)
```

#### ChartAnalysisResult
```python
@dataclass
class ChartAnalysisResult:
    """
    Result from chart data extraction and analysis
    """
    chart_type: str                         # Detected chart type
    title: str                              # Chart title
    data_points: List[Dict[str, Any]]       # Extracted data points
    trend_analysis: str                     # Trend description
    statistical_summary: Dict[str, Any]     # Statistical insights
    confidence_metrics: Dict[str, float]    # Confidence scores
    
    def get_data_as_dataframe(self) -> 'pandas.DataFrame':
        """Convert extracted data to pandas DataFrame"""
```

---

## Error Handling

### 🚨 Exception Classes

#### Custom Exceptions
```python
class RAGSystemError(Exception):
    """Base exception for RAG system errors"""
    
class DocumentProcessingError(RAGSystemError):
    """Raised when document processing fails"""
    
    def __init__(self, message: str, file_path: str = None, error_code: str = None):
        self.file_path = file_path
        self.error_code = error_code
        super().__init__(message)

class UnsupportedFormatError(DocumentProcessingError):
    """Raised when file format is not supported"""
    
class ExtractionError(DocumentProcessingError):
    """Raised when content extraction fails"""
    
class AIProcessingError(RAGSystemError):
    """Raised when AI model processing fails"""
    
class ConfigurationError(RAGSystemError):
    """Raised for configuration-related issues"""
```

#### Error Handling Patterns
```python
# Example error handling in client code
try:
    processor = DocumentProcessor()
    documents = processor.load_document("document.pdf")
    
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e.file_path}")
    # Handle unsupported format
    
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
    print(f"Error code: {e.error_code}")
    # Handle extraction failure with fallback
    
except AIProcessingError as e:
    print(f"AI processing failed: {e}")
    # Continue with non-AI processing
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

#### Error Recovery and Fallbacks
```python
class ErrorRecoveryMixin:
    """
    Mixin providing error recovery capabilities
    """
    
    def process_with_fallback(
        self,
        primary_method: callable,
        fallback_methods: List[callable],
        *args, **kwargs
    ):
        """
        Attempt processing with fallback methods
        
        Args:
            primary_method: Primary processing method
            fallback_methods: List of fallback methods
            
        Returns:
            Processing result or raises final exception
        """
        
        errors = []
        
        # Try primary method
        try:
            return primary_method(*args, **kwargs)
        except Exception as e:
            errors.append(("primary", str(e)))
        
        # Try fallback methods
        for i, method in enumerate(fallback_methods):
            try:
                result = method(*args, **kwargs)
                # Log successful fallback
                print(f"Fallback method {i+1} succeeded")
                return result
            except Exception as e:
                errors.append((f"fallback_{i+1}", str(e)))
        
        # All methods failed
        raise DocumentProcessingError(
            f"All processing methods failed: {errors}"
        )
```

---

## Usage Examples

### 💼 Complete Workflow Examples

#### End-to-End Document Processing
```python
import os
from src.document_loader import DocumentProcessor
from src.enhanced_rag import EnhancedRAG
from src.config import Config

def process_business_documents():
    """
    Complete workflow for processing business documents
    """
    
    # Initialize system with custom configuration
    config = Config()
    config.update_config({
        "PDF_EXTRACTION_METHODS": ["camelot", "pdfplumber"],
        "IMAGE_AI_ANALYSIS": True,
        "CONFIDENCE_THRESHOLD": 0.8
    })
    
    processor = DocumentProcessor(config)
    rag_system = EnhancedRAG(config)
    
    # Process multiple documents
    document_files = [
        "quarterly_report.pdf",
        "financial_data.xlsx", 
        "presentation_slides.pptx",
        "market_analysis.png"
    ]
    
    all_documents = []
    processing_results = {}
    
    for file_path in document_files:
        try:
            print(f"Processing {file_path}...")
            
            # Validate file first
            validation = processor.validate_file_support(file_path)
            if not validation['is_supported']:
                print(f"Skipping unsupported file: {file_path}")
                continue
            
            # Process document
            documents = processor.load_document(file_path)
            all_documents.extend(documents)
            
            # Get multimodal elements
            multimodal_elements = processor.get_multimodal_elements()
            
            # Store results
            processing_results[file_path] = {
                "documents": len(documents),
                "multimodal_elements": len(multimodal_elements),
                "tables": len([e for e in multimodal_elements if e.element_type == "table"]),
                "images": len([e for e in multimodal_elements if e.element_type == "image"])
            }
            
            print(f"✅ Processed {file_path}: {len(documents)} docs, {len(multimodal_elements)} elements")
            
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")
            processing_results[file_path] = {"error": str(e)}
    
    # Add all documents to RAG system
    if all_documents:
        rag_system.add_documents(all_documents)
        print(f"Added {len(all_documents)} documents to RAG system")
    
    # Example queries
    queries = [
        "What were the Q3 financial results?",
        "Show me revenue growth trends from the charts",
        "What are the key market insights?",
        "Compare performance metrics across all documents"
    ]
    
    print("\n🔍 Running example queries...")
    for query in queries:
        try:
            response = rag_system.query(
                query,
                include_multimodal=True,
                max_results=5,
                confidence_threshold=0.7
            )
            
            print(f"\nQuery: {query}")
            print(f"Answer: {response.answer[:200]}...")
            print(f"Sources: {len(response.source_documents)} documents")
            print(f"Multimodal elements: {len(response.multimodal_elements)}")
            print(f"Confidence: {response.confidence_score:.2f}")
            
        except Exception as e:
            print(f"Query failed: {e}")
    
    # Processing summary
    print(f"\n📊 Processing Summary:")
    total_docs = sum(r.get("documents", 0) for r in processing_results.values() if "error" not in r)
    total_elements = sum(r.get("multimodal_elements", 0) for r in processing_results.values() if "error" not in r)
    success_rate = len([r for r in processing_results.values() if "error" not in r]) / len(processing_results)
    
    print(f"Files processed: {len(document_files)}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Total documents: {total_docs}")
    print(f"Total multimodal elements: {total_elements}")
    
    return processing_results

if __name__ == "__main__":
    results = process_business_documents()
```

#### Advanced Multi-Modal Query Processing
```python
def advanced_multimodal_queries():
    """
    Examples of advanced multi-modal query capabilities
    """
    
    processor = DocumentProcessor()
    rag_system = EnhancedRAG()
    
    # Process a complex document
    documents = processor.load_document("financial_report_with_charts.pdf")
    rag_system.add_documents(documents)
    
    # Get multimodal elements for analysis
    multimodal_elements = processor.get_multimodal_elements()
    
    print(f"Processed document with {len(multimodal_elements)} multimodal elements")
    
    # Example 1: Table-specific query
    table_query = "What are the revenue figures by quarter from the financial tables?"
    response = rag_system.query(
        table_query,
        modalities=["tables", "text"],
        confidence_threshold=0.8
    )
    
    print(f"\n📊 Table Query: {table_query}")
    print(f"Answer: {response.answer}")
    print(f"Table sources: {len([e for e in response.multimodal_elements if e.element_type == 'table'])}")
    
    # Example 2: Image and chart analysis
    visual_query = "What trends are shown in the revenue growth chart?"
    response = rag_system.query(
        visual_query,
        modalities=["images", "charts", "text"],
        confidence_threshold=0.7
    )
    
    print(f"\n📈 Visual Query: {visual_query}")
    print(f"Answer: {response.answer}")
    print(f"Visual elements: {len([e for e in response.multimodal_elements if e.element_type in ['image', 'chart']])}")
    
    # Example 3: Cross-modal analysis
    cross_modal_query = "Compare the data in the revenue table with what's shown in the growth chart"
    response = rag_system.query(
        cross_modal_query,
        include_multimodal=True,
        max_results=8
    )
    
    print(f"\n🔗 Cross-Modal Query: {cross_modal_query}")
    print(f"Answer: {response.answer}")
    print(f"Sources used: Text={len(response.source_documents)}, Multimodal={len(response.multimodal_elements)}")
    
    # Example 4: Search multimodal content directly
    chart_elements = rag_system.search_multimodal_content(
        "revenue growth chart",
        element_types=["chart", "image"],
        min_confidence=0.8
    )
    
    print(f"\n🎯 Direct multimodal search found {len(chart_elements)} chart elements")
    for element in chart_elements:
        print(f"  - {element.element_type}: {element.text_description[:100]}... (confidence: {element.confidence_score:.2f})")

if __name__ == "__main__":
    advanced_multimodal_queries()
```

#### Batch Processing and Analytics
```python
def batch_processing_analytics():
    """
    Batch process multiple documents and generate analytics
    """
    
    processor = DocumentProcessor()
    
    # Process multiple files
    file_directory = "documents/"
    supported_files = []
    
    for filename in os.listdir(file_directory):
        file_path = os.path.join(file_directory, filename)
        if processor.validate_file(file_path):
            supported_files.append(file_path)
    
    print(f"Found {len(supported_files)} supported files for processing")
    
    # Batch process with error handling
    results = []
    
    for file_path in supported_files:
        try:
            start_time = time.time()
            
            # Process document
            documents = processor.load_document(file_path)
            multimodal_elements = processor.get_multimodal_elements()
            
            processing_time = time.time() - start_time
            
            # Analyze processing results
            result = {
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "success": True,
                "processing_time": processing_time,
                "documents_created": len(documents),
                "multimodal_elements": len(multimodal_elements),
                "tables": len([e for e in multimodal_elements if e.element_type == "table"]),
                "images": len([e for e in multimodal_elements if e.element_type == "image"]),
                "charts": len([e for e in multimodal_elements if e.element_type == "chart"]),
                "avg_confidence": sum(e.confidence_score for e in multimodal_elements) / len(multimodal_elements) if multimodal_elements else 0
            }
            
            results.append(result)
            print(f"✅ {result['filename']}: {result['documents_created']} docs, {result['multimodal_elements']} elements")
            
        except Exception as e:
            results.append({
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "success": False,
                "error": str(e)
            })
            print(f"❌ {os.path.basename(file_path)}: {e}")
    
    # Generate analytics
    successful_results = [r for r in results if r["success"]]
    
    analytics = {
        "total_files": len(results),
        "successful_files": len(successful_results),
        "success_rate": len(successful_results) / len(results) if results else 0,
        "total_documents": sum(r["documents_created"] for r in successful_results),
        "total_multimodal_elements": sum(r["multimodal_elements"] for r in successful_results),
        "avg_processing_time": sum(r["processing_time"] for r in successful_results) / len(successful_results) if successful_results else 0,
        "avg_confidence": sum(r["avg_confidence"] for r in successful_results) / len(successful_results) if successful_results else 0
    }
    
    # Print analytics
    print(f"\n📊 Batch Processing Analytics:")
    print(f"Files processed: {analytics['total_files']}")
    print(f"Success rate: {analytics['success_rate']:.1%}")
    print(f"Total documents created: {analytics['total_documents']}")
    print(f"Total multimodal elements: {analytics['total_multimodal_elements']}")
    print(f"Average processing time: {analytics['avg_processing_time']:.2f} seconds")
    print(f"Average confidence score: {analytics['avg_confidence']:.2f}")
    
    # File type breakdown
    file_types = {}
    for result in successful_results:
        ext = os.path.splitext(result["filename"])[1].lower()
        if ext not in file_types:
            file_types[ext] = {"count": 0, "elements": 0}
        file_types[ext]["count"] += 1
        file_types[ext]["elements"] += result["multimodal_elements"]
    
    print(f"\n📁 File Type Analysis:")
    for ext, stats in file_types.items():
        print(f"  {ext}: {stats['count']} files, {stats['elements']} total elements")
    
    return results, analytics

if __name__ == "__main__":
    results, analytics = batch_processing_analytics()
```

---

## Best Practices

### 🎯 API Usage Best Practices

#### Performance Optimization
```python
# ✅ Good practices for performance

# 1. Reuse processor instances
processor = DocumentProcessor()  # Initialize once
rag_system = EnhancedRAG()      # Initialize once

# Process multiple documents with the same instance
for file_path in file_paths:
    documents = processor.load_document(file_path)
    rag_system.add_documents(documents)

# 2. Batch document processing
documents = processor.load_multiple_documents(
    file_paths,
    parallel=True,        # Enable parallel processing
    max_workers=3         # Optimal for most systems
)

# 3. Use appropriate confidence thresholds
response = rag_system.query(
    query,
    confidence_threshold=0.7,  # Balance quality vs coverage
    max_results=5              # Limit results for faster response
)

# 4. Clear cache periodically for long-running processes
processor.clear_multimodal_cache()  # Free memory
```

#### Error Handling Best Practices
```python
# ✅ Robust error handling

def robust_document_processing(file_paths):
    processor = DocumentProcessor()
    successful_docs = []
    failed_files = []
    
    for file_path in file_paths:
        try:
            # Validate before processing
            validation = processor.validate_file_support(file_path)
            if not validation['is_supported']:
                print(f"Skipping unsupported file: {file_path}")
                continue
            
            # Process with timeout handling
            documents = processor.load_document(file_path)
            successful_docs.extend(documents)
            
        except UnsupportedFormatError:
            print(f"Format not supported: {file_path}")
            failed_files.append((file_path, "unsupported_format"))
            
        except DocumentProcessingError as e:
            print(f"Processing failed: {file_path} - {e}")
            failed_files.append((file_path, str(e)))
            
        except Exception as e:
            print(f"Unexpected error: {file_path} - {e}")
            failed_files.append((file_path, f"unexpected_error: {e}"))
    
    return successful_docs, failed_files
```

#### Memory Management
```python
# ✅ Efficient memory usage

def process_large_document_collection(file_paths):
    processor = DocumentProcessor()
    rag_system = EnhancedRAG()
    
    # Process in batches to manage memory
    batch_size = 5
    
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        
        # Process batch
        batch_documents = []
        for file_path in batch:
            try:
                documents = processor.load_document(file_path)
                batch_documents.extend(documents)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Add to RAG system
        if batch_documents:
            rag_system.add_documents(batch_documents)
        
        # Clear processor cache between batches
        processor.clear_multimodal_cache()
        
        print(f"Processed batch {i//batch_size + 1}: {len(batch_documents)} documents")
```

#### Configuration Best Practices
```python
# ✅ Optimal configuration practices

# 1. Environment-specific configurations
def get_optimized_config(environment: str) -> Config:
    config = Config()
    
    if environment == "development":
        config.update_config({
            "PDF_EXTRACTION_METHODS": ["pdfplumber"],  # Fast for development
            "IMAGE_AI_ANALYSIS": False,                # Skip AI for speed
            "PARALLEL_PROCESSING": False               # Easier debugging
        })
    
    elif environment == "production":
        config.update_config({
            "PDF_EXTRACTION_METHODS": ["camelot", "pdfplumber"],  # High accuracy
            "IMAGE_AI_ANALYSIS": True,                            # Full AI features
            "PARALLEL_PROCESSING": True,                          # Maximum performance
            "CONFIDENCE_THRESHOLD": 0.8                           # High quality threshold
        })
    
    return config

# 2. Hardware-specific optimization
def optimize_for_system():
    config = Config()
    
    # Check available resources
    import psutil
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = psutil.cpu_count()
    
    if ram_gb >= 16:
        config.update_config({
            "PARALLEL_PROCESSING": True,
            "MAX_WORKERS": min(4, cpu_cores),
            "CHUNK_SIZE": 1200
        })
    else:
        config.update_config({
            "PARALLEL_PROCESSING": False,
            "CHUNK_SIZE": 800,
            "ENABLE_CACHING": False  # Reduce memory usage
        })
    
    return config
```

---

## 📞 Support and Resources

### 🔗 Additional Resources

- **GitHub Repository**: [https://github.com/fenilsonani/rag-document-qa](https://github.com/fenilsonani/rag-document-qa)
- **Issue Tracker**: Report bugs and request features
- **Documentation**: Comprehensive guides and tutorials
- **Examples**: Real-world usage examples and templates

### 💡 Tips for Success

1. **Start Simple**: Begin with basic document processing before adding multimodal features
2. **Validate Early**: Always validate files before processing to avoid errors
3. **Monitor Performance**: Track processing times and memory usage
4. **Use Confidence Scores**: Filter results based on confidence for better quality
5. **Handle Errors Gracefully**: Implement comprehensive error handling
6. **Batch Processing**: Process multiple documents efficiently
7. **Regular Cleanup**: Clear caches and temporary files regularly

---

**🚀 Ready to build powerful document intelligence applications!**

**Built with 💙 by [Fenil Sonani](https://github.com/fenilsonani) | © 2025 | API Documentation v2.0**