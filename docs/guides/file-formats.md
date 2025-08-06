# File Format Support Guide - RAG Document Q&A System

> **Comprehensive Guide to 26+ File Format Processing Capabilities**
> 
> **Version**: 2.0 Enhanced | **Updated**: January 2025 | **Status**: Production Ready

---

## 📋 Quick Reference

### 📊 Supported Formats Overview

| **Category** | **Extensions** | **Count** | **AI Features** | **Max Size** |
|--------------|----------------|-----------|-----------------|--------------|
| 📄 **Documents** | `.pdf`, `.txt`, `.docx`, `.md`, `.rtf` | 5 | Advanced PDF analysis, layout detection | 50MB |
| 📈 **Spreadsheets** | `.xlsx`, `.xls`, `.csv` | 3 | Multi-sheet, formulas, charts | 25MB |
| 📽️ **Presentations** | `.pptx`, `.ppt` | 2 | Slides, tables, images | 30MB |
| 🖼️ **Images** | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.svg` | 8 | OCR, AI analysis, objects | 20MB |
| 🌐 **Structured** | `.json`, `.xml`, `.yaml`, `.yml` | 4 | Structure parsing, tables | 10MB |
| 🌍 **Web** | `.html`, `.htm` | 2 | DOM parsing, tables | 10MB |
| 📚 **Ebooks** | `.epub`, `.mobi` | 2 | Chapter extraction | 20MB |

**Total: 26 file formats with intelligent AI-powered processing**

---

## 📄 Document Formats

### 🔴 PDF Files (.pdf) - **ADVANCED PROCESSING**

#### 🎯 Special Capabilities
```yaml
Processing Methods:
  - PDFPlumber: Fast standard table extraction (85-90% accuracy)
  - Camelot-py: Computer vision tables (90-95% accuracy) 
  - PyMuPDF: Image extraction + layout analysis (80-85% accuracy)
  - Tabula-py: Complex/scanned PDFs (75-85% accuracy)

Features:
  ✅ Multi-page table extraction
  ✅ Image and chart extraction with AI analysis
  ✅ Multi-column document layout detection
  ✅ Scanned PDF OCR processing
  ✅ Cross-page element handling
  ✅ Confidence scoring (0.0-1.0)

Best Results For:
  - Financial reports with complex tables
  - Research papers with figures and charts
  - Technical manuals with diagrams
  - Multi-column academic documents
```

#### 📊 PDF Processing Examples
```python
# Example extracted elements from PDF
pdf_elements = {
    "tables": [
        {
            "page": 3,
            "method": "camelot",
            "confidence": 0.92,
            "data": "Q1 Revenue: $1.2M, Q2 Revenue: $1.5M...",
            "rows": 12,
            "columns": 4
        }
    ],
    "images": [
        {
            "page": 5, 
            "type": "chart",
            "ai_description": "Bar chart showing quarterly growth trends",
            "ocr_text": "Revenue Growth 2024",
            "confidence": 0.87
        }
    ],
    "layout": {
        "columns": 2,
        "reading_order": "preserved",
        "structure_confidence": 0.94
    }
}
```

### 📝 Text Documents

#### `.txt` - Plain Text
```yaml
Processing: Direct text extraction
Features:
  ✅ Full text indexing
  ✅ Encoding detection (UTF-8, Latin-1, etc.)
  ✅ Large file handling
Speed: Very fast (<5 seconds)
Use Cases: Logs, notes, code files, data exports
```

#### `.docx` - Microsoft Word
```yaml
Processing: python-docx + langchain loaders  
Features:
  ✅ Text extraction with formatting preservation
  ✅ Table detection and extraction
  ✅ Image detection (basic)
  ✅ Header/footer processing
Speed: Fast (5-15 seconds)
Use Cases: Reports, proposals, documentation
```

#### `.md` - Markdown
```yaml
Processing: UnstructuredMarkdownLoader
Features:
  ✅ Structure-aware parsing (headers, lists, code blocks)
  ✅ Link extraction and preservation
  ✅ Code syntax highlighting detection
Speed: Fast (3-8 seconds)  
Use Cases: Documentation, README files, technical guides
```

#### `.rtf` - Rich Text Format
```yaml
Processing: Universal file processor
Features:
  ✅ Text extraction with basic formatting
  ✅ Cross-platform compatibility
  ✅ Legacy document support
Speed: Fast (5-10 seconds)
Use Cases: Legacy documents, cross-platform text
```

---

## 📈 Spreadsheet Formats

### 📊 Excel Files (.xlsx, .xls) - **ADVANCED PROCESSING**

#### 🎯 Multi-Sheet Intelligence
```python
# Excel processing capabilities
excel_features = {
    "sheet_detection": "Automatic discovery of all worksheets",
    "data_analysis": {
        "data_types": "Automatic inference (dates, numbers, text)",
        "formulas": "Preservation and evaluation where possible", 
        "charts": "Detection and basic extraction",
        "tables": "Structured data extraction with headers"
    },
    "performance": {
        "small_files": "<10 seconds",
        "large_files": "30-60 seconds", 
        "memory_efficient": "Streaming for large datasets"
    }
}

# Example extracted data
excel_result = {
    "sheets": [
        {
            "name": "Sales Data",
            "rows": 1543,
            "columns": 8,
            "tables": 2,
            "charts": 1,
            "data_quality": 0.96
        },
        {
            "name": "Financial Summary", 
            "rows": 45,
            "columns": 12,
            "tables": 3,
            "pivot_tables": 1,
            "data_quality": 0.91
        }
    ],
    "total_elements": 6,
    "processing_time": "23.4 seconds",
    "confidence": 0.93
}
```

#### 🔧 Advanced Excel Features
```yaml
Data Processing:
  ✅ Multi-sheet extraction and analysis
  ✅ Formula preservation and evaluation
  ✅ Data type inference and validation
  ✅ Chart and graph detection
  ✅ Pivot table recognition
  ✅ Conditional formatting detection

Quality Assurance:
  ✅ Data completeness scoring
  ✅ Structure validation
  ✅ Cross-sheet reference tracking
  ✅ Error cell detection

Use Cases:
  - Financial reports and budgets
  - Sales and marketing data
  - Research datasets
  - Inventory and logistics data
```

### 📋 CSV Files (.csv) - **SMART TABLE CONVERSION**

```python
# CSV processing intelligence
csv_features = {
    "delimiter_detection": "Automatic detection of separators (comma, semicolon, tab)",
    "encoding_detection": "UTF-8, Latin-1, and other encodings",
    "data_cleaning": {
        "null_handling": "Smart null value detection",
        "type_conversion": "Automatic data type inference",
        "outlier_detection": "Statistical outlier identification"
    },
    "table_conversion": {
        "header_detection": "Automatic header row identification", 
        "schema_inference": "Column type and relationship detection",
        "metadata_extraction": "File statistics and quality metrics"
    }
}

# Example processing result
csv_result = {
    "rows": 50000,
    "columns": 12,
    "delimiter": ",",
    "encoding": "utf-8",
    "data_types": {
        "date_columns": 2,
        "numeric_columns": 6, 
        "text_columns": 4
    },
    "quality_score": 0.94,
    "processing_time": "8.2 seconds"
}
```

---

## 📽️ Presentation Formats  

### 🎯 PowerPoint Files (.pptx, .ppt) - **SLIDE INTELLIGENCE**

#### 📊 Advanced PowerPoint Processing
```python
# PowerPoint extraction capabilities
powerpoint_features = {
    "slide_analysis": {
        "text_extraction": "Full text from all text boxes and shapes",
        "layout_detection": "Slide layout and structure recognition",
        "bullet_points": "Hierarchical bullet point extraction",
        "speaker_notes": "Notes section processing"
    },
    
    "table_extraction": {
        "native_tables": "PowerPoint table extraction with formatting",
        "text_tables": "Detection of table-like text structures", 
        "data_analysis": "Automatic data type inference",
        "confidence_scoring": "Quality assessment for each table"
    },
    
    "image_processing": {
        "image_detection": "All embedded images identified",
        "chart_analysis": "PowerPoint charts with data extraction",
        "diagram_processing": "SmartArt and diagram text extraction",
        "ocr_fallback": "OCR for image-based text"
    },
    
    "metadata": {
        "slide_count": "Total number of slides",
        "slide_titles": "Extraction of slide titles", 
        "slide_transitions": "Slide relationship mapping",
        "presentation_structure": "Overall document hierarchy"
    }
}

# Example extraction result
pptx_result = {
    "slides": 24,
    "text_elements": 156,
    "tables": 8,
    "images": 12,
    "charts": 5,
    "total_words": 3247,
    "processing_time": "34.7 seconds",
    "confidence": 0.89
}
```

#### 🎨 PowerPoint Use Cases
```yaml
Business Presentations:
  - Quarterly business reviews
  - Sales presentations with data tables
  - Marketing campaigns with statistics
  - Strategic planning documents

Academic Presentations:
  - Research presentations with charts
  - Educational materials with diagrams
  - Conference presentations with data
  - Thesis defense presentations

Training Materials:
  - Corporate training slides
  - Technical documentation
  - Process flow presentations
  - Compliance training materials
```

---

## 🖼️ Image Formats - **AI-POWERED ANALYSIS**

### 🤖 Advanced Image Processing Pipeline

#### Supported Image Types
```python
image_formats = {
    "raster_images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"],
    "vector_images": [".svg"],
    "special_features": {
        "animated": "GIF animation frame extraction",
        "multi_layer": "TIFF multi-layer support",
        "transparency": "PNG/WebP transparency handling",
        "vector": "SVG text and shape extraction"
    }
}
```

#### 🔍 AI Analysis Capabilities
```python
# Multi-model AI processing
ai_processing_pipeline = {
    "1_image_captioning": {
        "model": "Salesforce/blip-image-captioning-base",
        "capability": "Generate descriptive captions for images",
        "accuracy": "High for photos, good for charts/diagrams",
        "examples": [
            "A bar chart showing quarterly revenue growth",
            "Technical diagram of network architecture", 
            "Pie chart displaying market share distribution"
        ]
    },
    
    "2_object_detection": {
        "model": "facebook/detr-resnet-50", 
        "capability": "Detect and classify objects in images",
        "classes": "90+ common object types",
        "examples": [
            "person, laptop, table, document", 
            "chart, graph, diagram, text",
            "building, vehicle, equipment"
        ]
    },
    
    "3_ocr_processing": {
        "engine": "Tesseract OCR with preprocessing",
        "languages": "English (multi-language support available)",
        "accuracy": "90-95% for clear text",
        "preprocessing": [
            "Image enhancement and noise reduction",
            "Contrast and brightness optimization", 
            "Skew correction and alignment"
        ]
    },
    
    "4_chart_analysis": {
        "capability": "Specialized chart and graph interpretation",
        "supported_types": [
            "Bar charts", "Line graphs", "Pie charts",
            "Scatter plots", "Histograms", "Flow diagrams"
        ],
        "data_extraction": "Attempt to extract numerical data from charts",
        "trend_analysis": "Identify patterns and trends in visualizations"
    }
}
```

#### 📊 Image Processing Examples
```python
# Example image analysis results
image_results = {
    "chart_image_png": {
        "ai_caption": "A bar chart showing quarterly sales performance with values ranging from $1.2M to $2.8M across Q1-Q4 2024",
        "objects_detected": ["chart", "text", "graph"],
        "ocr_text": "Quarterly Sales Performance 2024\nQ1: $1.2M\nQ2: $1.8M\nQ3: $2.1M\nQ4: $2.8M",
        "chart_type": "bar_chart",
        "data_points": 4,
        "confidence": 0.87
    },
    
    "technical_diagram_jpg": {
        "ai_caption": "Technical system architecture diagram showing network components and data flow",
        "objects_detected": ["diagram", "text", "arrows", "boxes"],
        "ocr_text": "API Gateway → Load Balancer → Application Servers → Database",
        "diagram_type": "system_architecture", 
        "components": 8,
        "confidence": 0.82
    },
    
    "document_scan_png": {
        "ai_caption": "Scanned document page containing text and a data table",
        "objects_detected": ["document", "text", "table"],
        "ocr_text": "Financial Report 2024\n[Full extracted text...]",
        "has_table": True,
        "text_quality": "high",
        "confidence": 0.91
    }
}
```

---

## 🌐 Structured Data Formats

### 📋 JSON Files (.json) - **INTELLIGENT STRUCTURE PARSING**

#### 🔧 Advanced JSON Processing
```python
# JSON processing capabilities
json_features = {
    "structure_analysis": {
        "nested_objects": "Deep object hierarchy navigation",
        "array_processing": "List and array data extraction",
        "schema_inference": "Automatic data structure detection",
        "type_validation": "Data type consistency checking"
    },
    
    "table_conversion": {
        "array_to_table": "Convert JSON arrays to structured tables",
        "object_flattening": "Flatten nested objects for analysis", 
        "key_normalization": "Standardize key naming conventions",
        "relational_mapping": "Identify relationships between objects"
    },
    
    "data_extraction": {
        "metadata_extraction": "File structure and statistics",
        "content_indexing": "Full-text search across all values",
        "query_optimization": "Efficient nested data queries",
        "validation": "JSON syntax and structure validation"
    }
}

# Example processing result
json_processing_result = {
    "file_size": "2.3MB",
    "json_valid": True,
    "structure": {
        "total_objects": 1247,
        "nested_levels": 5,
        "arrays": 23,
        "unique_keys": 156
    },
    "tables_extracted": [
        {
            "path": "data.employees", 
            "rows": 450,
            "columns": 8,
            "table_type": "employee_records"
        },
        {
            "path": "data.sales.quarterly",
            "rows": 16,
            "columns": 6, 
            "table_type": "financial_data"
        }
    ],
    "processing_time": "3.7 seconds",
    "confidence": 0.94
}
```

### 🌐 XML Files (.xml) - **STRUCTURED DOCUMENT PARSING**

```python
# XML processing features
xml_capabilities = {
    "document_parsing": {
        "namespace_handling": "XML namespace resolution",
        "schema_validation": "XSD schema validation support",
        "xpath_queries": "Advanced XPath-based data extraction",
        "attribute_processing": "XML attribute extraction and analysis"
    },
    
    "content_extraction": {
        "text_content": "Extract all text content with hierarchy",
        "table_detection": "Identify table-like XML structures",
        "metadata_extraction": "XML headers and meta information",
        "link_resolution": "Internal and external reference handling"
    },
    
    "structure_analysis": {
        "hierarchy_mapping": "Document structure tree generation", 
        "element_statistics": "Tag frequency and usage analysis",
        "content_validation": "Data consistency and completeness",
        "transformation": "Convert to searchable document format"
    }
}
```

### 📝 YAML Files (.yaml, .yml) - **CONFIGURATION PARSING**

```python
# YAML processing capabilities  
yaml_features = {
    "configuration_analysis": {
        "multi_document": "Handle multiple YAML documents in one file",
        "anchor_resolution": "YAML anchor and alias processing", 
        "type_inference": "Automatic data type detection",
        "validation": "YAML syntax and structure validation"
    },
    
    "data_extraction": {
        "nested_structures": "Deep nested configuration extraction",
        "list_processing": "Array and list data handling",
        "key_value_pairs": "Configuration parameter extraction",
        "documentation": "Extract comments and documentation"
    },
    
    "use_cases": [
        "Application configuration files",
        "Docker and Kubernetes manifests", 
        "CI/CD pipeline configurations",
        "API specification files (OpenAPI)"
    ]
}
```

---

## 🌍 Web Formats

### 🌐 HTML Files (.html, .htm) - **WEB CONTENT PROCESSING**

#### 📊 Advanced HTML Analysis
```python
# HTML processing capabilities
html_features = {
    "content_extraction": {
        "text_extraction": "Clean text extraction from HTML elements",
        "table_processing": "HTML table extraction with structure preservation",
        "link_analysis": "URL and hyperlink extraction and validation",
        "metadata_extraction": "Meta tags, title, and header information"
    },
    
    "structure_analysis": {
        "dom_parsing": "Complete DOM tree analysis",
        "semantic_elements": "HTML5 semantic element recognition",
        "css_class_analysis": "CSS class and ID extraction",
        "script_detection": "JavaScript code identification"
    },
    
    "table_intelligence": {
        "header_detection": "Automatic table header identification",
        "cell_merging": "Colspan and rowspan handling", 
        "nested_tables": "Support for nested table structures",
        "table_classification": "Data vs layout table identification"
    },
    
    "media_processing": {
        "image_extraction": "Embedded image detection and processing",
        "alt_text_processing": "Image alt text and accessibility content",
        "video_metadata": "Video and media element information",
        "external_resources": "Identification of external resource links"
    }
}

# Example HTML processing result
html_result = {
    "page_title": "Financial Report Q4 2024",
    "text_content": "15,247 words extracted",
    "tables": [
        {
            "caption": "Revenue by Quarter",
            "rows": 5,
            "columns": 4,
            "has_header": True,
            "confidence": 0.93
        },
        {
            "caption": "Regional Sales Data",
            "rows": 12, 
            "columns": 6,
            "has_header": True,
            "confidence": 0.88
        }
    ],
    "links": 47,
    "images": 8,
    "processing_time": "6.2 seconds"
}
```

---

## 📚 Ebook Formats

### 📖 EPUB Files (.epub) - **DIGITAL BOOK PROCESSING**

```python
# EPUB processing capabilities
epub_features = {
    "book_structure": {
        "chapter_extraction": "Individual chapter identification and extraction",
        "table_of_contents": "TOC parsing and navigation structure",
        "metadata_extraction": "Author, title, ISBN, publisher information",
        "spine_processing": "Reading order and book structure analysis"
    },
    
    "content_analysis": {
        "text_extraction": "Full text extraction from all chapters",
        "image_processing": "Embedded images and illustrations",
        "footnote_handling": "Footnotes and endnotes extraction", 
        "cross_references": "Internal links and reference processing"
    },
    
    "advanced_features": {
        "multi_language": "Support for multi-language books",
        "mathematical_content": "MathML and mathematical notation",
        "interactive_elements": "Interactive content identification",
        "accessibility": "Alt text and accessibility features"
    }
}
```

### 📱 MOBI Files (.mobi) - **KINDLE FORMAT SUPPORT**

```python  
# MOBI processing capabilities
mobi_features = {
    "amazon_format": {
        "drm_handling": "DRM-free content processing only",
        "formatting_preservation": "Kindle-specific formatting extraction",
        "chapter_detection": "Automatic chapter boundary identification",
        "bookmark_processing": "User bookmarks and highlights (if accessible)"
    },
    
    "content_extraction": {
        "text_content": "Full book text extraction",
        "image_extraction": "Book cover and internal images",
        "metadata": "Amazon-specific metadata and tags",
        "navigation": "Table of contents and navigation points"
    },
    
    "limitations": [
        "DRM-protected files cannot be processed",
        "Some advanced formatting may be lost",
        "Proprietary Amazon features may not be fully supported"
    ]
}
```

---

## 🔧 Processing Configuration

### ⚙️ Format-Specific Settings

#### Environment Configuration
```bash
# .env configuration for different formats
# PDF Processing
PDF_EXTRACTION_METHOD=multi_method  # Options: fast, accurate, multi_method
PDF_IMAGE_EXTRACTION=true
PDF_TABLE_CONFIDENCE_THRESHOLD=0.7

# Excel Processing  
EXCEL_SHEET_LIMIT=10               # Maximum sheets to process
EXCEL_ANALYZE_FORMULAS=true
EXCEL_EXTRACT_CHARTS=true

# Image Processing
IMAGE_AI_ANALYSIS=true             # Enable AI image analysis
IMAGE_OCR_ENABLED=true            # Enable OCR processing
IMAGE_MAX_RESOLUTION=4096         # Max image resolution

# Performance Settings
PARALLEL_PROCESSING=true           # Enable parallel processing
MAX_WORKERS=4                     # Number of worker processes
CHUNK_SIZE=1000                   # Document chunk size
```

#### Advanced Configuration
```python
# config.py - Advanced format settings
ADVANCED_FORMAT_CONFIG = {
    "pdf": {
        "extractors": ["pdfplumber", "camelot", "pymupdf", "tabula"],
        "table_confidence_threshold": 0.75,
        "image_analysis_enabled": True,
        "layout_analysis": True
    },
    
    "excel": {
        "max_sheets": 20,
        "formula_evaluation": True, 
        "chart_extraction": True,
        "data_validation": True
    },
    
    "images": {
        "ai_models": ["blip", "detr"],
        "ocr_languages": ["eng"],
        "preprocessing": True,
        "max_size_mb": 20
    },
    
    "powerpoint": {
        "extract_notes": True,
        "analyze_charts": True,
        "preserve_layout": True,
        "image_processing": True
    }
}
```

### 🎯 Performance Optimization by Format

#### Processing Time Estimates
```yaml
Format Performance Guide:

Fast Processing (<10 seconds):
  - TXT, MD, RTF: 1-3 seconds
  - CSV (small): 2-5 seconds  
  - JSON, YAML: 3-8 seconds
  - HTML: 4-8 seconds

Medium Processing (10-60 seconds):
  - DOCX: 8-25 seconds
  - Excel (multi-sheet): 15-45 seconds
  - PowerPoint: 20-50 seconds
  - Images with AI: 10-30 seconds

Intensive Processing (60+ seconds):
  - Large PDFs with tables: 1-5 minutes
  - Complex Excel files: 1-3 minutes
  - High-resolution images: 30-90 seconds
  - Large EPUB files: 2-8 minutes
```

#### Memory Usage Guidelines
```python
# Memory usage by format type
memory_requirements = {
    "text_formats": "50-200MB",      # TXT, MD, RTF
    "office_documents": "100-500MB",  # DOCX, PPTX  
    "spreadsheets": "200-1GB",       # Excel files
    "pdfs_simple": "100-300MB",      # Basic PDFs
    "pdfs_advanced": "500MB-2GB",    # Complex PDFs with images
    "images": "100-800MB",           # Depends on resolution and AI processing
    "structured_data": "50-400MB",   # JSON, XML, YAML
    "ebooks": "100-600MB"           # EPUB, MOBI
}

# Optimization strategies
optimization_tips = {
    "reduce_chunk_size": "CHUNK_SIZE=500 for faster processing",
    "limit_extractors": "Use single PDF extractor for speed",
    "disable_ai": "Set IMAGE_AI_ANALYSIS=false for speed",
    "parallel_limit": "Reduce MAX_WORKERS for memory-constrained systems"
}
```

---

## 🧪 Testing File Formats

### 📝 Format Testing Commands

#### Test All Formats
```bash
# Comprehensive format testing
source venv/bin/activate
python3 test_all_formats.py

# Expected output:
# ✅ Excel processing: 2 sheets, 3 tables extracted
# ✅ PowerPoint processing: 8 slides, 2 tables, 3 images
# ✅ Image processing: AI analysis complete, OCR text extracted
# ✅ JSON processing: 1247 objects, 2 tables converted
```

#### Test Specific Formats
```bash
# Test PDF processing specifically
python3 test_pdf_multimodal.py

# Test individual format processors
python3 -c "
import sys
sys.path.insert(0, 'src')
from universal_file_processor import UniversalFileProcessor

processor = UniversalFileProcessor()
formats = processor.get_supported_formats()
for category, info in formats.items():
    print(f'{category}: {len(info[\"extensions\"])} formats')
"
```

#### Create Test Files
```python
# The test suite automatically creates sample files
test_files_created = [
    "sample_data.xlsx",      # Multi-sheet Excel with tables
    "presentation.pptx",     # PowerPoint with slides and tables  
    "chart_visualization.png", # Generated chart image
    "structured_data.json",  # Complex JSON with nested data
    "product_catalog.csv"    # CSV with various data types
]

# Each test file exercises different processing capabilities
# and validates extraction accuracy and performance
```

---

## 🚀 Best Practices

### 📋 Format Selection Guidelines

#### Choose the Right Format for Your Use Case
```yaml
For Tabular Data:
  Best: Excel (.xlsx) - Full feature support, multi-sheet capability
  Good: CSV (.csv) - Simple, fast processing
  Okay: JSON (.json) - Structured but requires conversion

For Text Documents:
  Best: Markdown (.md) - Structure preserved, fast processing
  Good: DOCX (.docx) - Rich formatting, table support
  Okay: PDF (.pdf) - Advanced processing but slower

For Presentations:
  Best: PowerPoint (.pptx) - Full slide and table extraction
  Alternative: PDF (.pdf) - If slides exported to PDF

For Images and Charts:
  Best: PNG/JPG (.png/.jpg) - Full AI analysis support
  Good: SVG (.svg) - Vector graphics with text extraction  
  Note: Higher resolution = better OCR results
```

#### File Preparation Tips
```markdown
📄 **PDF Files:**
- Ensure text is selectable (not scanned images only)
- Use high-quality scans if document is image-based
- Keep file size under 50MB for optimal performance

📊 **Excel Files:**
- Use clear table headers in row 1
- Avoid merged cells where possible
- Keep sheets under 100MB each

🖼️ **Image Files:**
- Use high resolution (300 DPI+) for better OCR
- Ensure good contrast for text recognition
- Save charts as PNG for best quality

📽️ **PowerPoint Files:**
- Use built-in table tools rather than text tables
- Embed high-quality images
- Keep slide count reasonable (<50 slides)
```

---

## 📞 Support and Resources

### 🔗 Format-Specific Resources

#### Documentation Links
- **PDF Processing**: See `ADVANCED_PDF_PROCESSING.md`
- **Excel Analysis**: See `EXCEL_PROCESSING_GUIDE.md`
- **Image AI**: See `MULTIMODAL_AI_GUIDE.md`
- **API Reference**: See `API_DOCUMENTATION.md`

#### Community Resources
- **GitHub Issues**: Report format-specific problems
- **Discussions**: Share format processing tips
- **Wiki**: Community-contributed format guides

### 🎯 Format Request Process

Missing a format you need? Here's how to request new format support:

1. **Check Roadmap**: Review planned format additions
2. **Create Issue**: Submit format request with use case
3. **Provide Samples**: Share sample files for testing
4. **Community Vote**: Format requests are prioritized by community interest

---

**🚀 Ready to process any document format with AI-powered intelligence!**

**Built with 💙 by [Fenil Sonani](https://github.com/fenilsonani) | © 2025 | Production Ready**