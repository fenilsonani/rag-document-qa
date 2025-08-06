"""
Unit tests for advanced_pdf_processor.py module.
Tests PDF processing, table extraction, image extraction, and layout analysis functions.
"""

import pytest
import sys
import os
import tempfile
import io
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock optional dependencies before importing
with patch.dict('sys.modules', {
    'pdfplumber': MagicMock(),
    'camelot': MagicMock(),
    'fitz': MagicMock(),
    'tabula': MagicMock()
}):
    from src.advanced_pdf_processor import (
        AdvancedPDFProcessor, PDFLayoutElement, PDFPageAnalysis
    )
    from src.config import Config
    from src.multimodal_rag import MultiModalElement
    from langchain.schema import Document


@pytest.fixture
def pdf_processor():
    """Create AdvancedPDFProcessor instance for testing."""
    with patch('src.advanced_pdf_processor.Config') as mock_config:
        mock_config.return_value = Mock()
        return AdvancedPDFProcessor()


@pytest.fixture
def sample_pdf_path():
    """Create a temporary PDF file path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(b'%PDF-1.4\n%dummy pdf content')
        return tmp_file.name


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for table testing."""
    return pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Tokyo']
    })


@pytest.fixture
def sample_layout_element():
    """Create sample PDFLayoutElement for testing."""
    return PDFLayoutElement(
        element_id="test_element_1",
        element_type="text",
        page_number=1,
        bbox=(10, 10, 100, 50),
        content="Sample text content",
        metadata={"font_size": 12},
        confidence_score=0.9,
        extraction_method="test"
    )


class TestPDFLayoutElement:
    """Test PDFLayoutElement dataclass."""
    
    def test_layout_element_creation(self, sample_layout_element):
        """Test PDFLayoutElement creation."""
        element = sample_layout_element
        
        assert element.element_id == "test_element_1"
        assert element.element_type == "text"
        assert element.page_number == 1
        assert element.bbox == (10, 10, 100, 50)
        assert element.content == "Sample text content"
        assert element.confidence_score == 0.9
    
    def test_get_area(self, sample_layout_element):
        """Test area calculation."""
        element = sample_layout_element
        area = element.get_area()
        
        # (100-10) * (50-10) = 90 * 40 = 3600
        assert area == 3600.0
    
    def test_get_area_zero_dimension(self):
        """Test area calculation with zero dimensions."""
        element = PDFLayoutElement(
            element_id="zero_element",
            element_type="text",
            page_number=1,
            bbox=(10, 10, 10, 20),  # Zero width
            content="Zero width",
            metadata={}
        )
        
        assert element.get_area() == 0.0
    
    def test_overlaps_with_overlapping(self):
        """Test overlap detection with overlapping elements."""
        element1 = PDFLayoutElement(
            element_id="elem1", element_type="text", page_number=1,
            bbox=(10, 10, 50, 50), content="Text 1", metadata={}
        )
        element2 = PDFLayoutElement(
            element_id="elem2", element_type="text", page_number=1,
            bbox=(30, 30, 70, 70), content="Text 2", metadata={}
        )
        
        assert element1.overlaps_with(element2)
        assert element2.overlaps_with(element1)
    
    def test_overlaps_with_non_overlapping(self):
        """Test overlap detection with non-overlapping elements."""
        element1 = PDFLayoutElement(
            element_id="elem1", element_type="text", page_number=1,
            bbox=(10, 10, 30, 30), content="Text 1", metadata={}
        )
        element2 = PDFLayoutElement(
            element_id="elem2", element_type="text", page_number=1,
            bbox=(50, 50, 70, 70), content="Text 2", metadata={}
        )
        
        assert not element1.overlaps_with(element2)
        assert not element2.overlaps_with(element1)
    
    def test_overlaps_with_custom_threshold(self):
        """Test overlap detection with custom threshold."""
        element1 = PDFLayoutElement(
            element_id="elem1", element_type="text", page_number=1,
            bbox=(10, 10, 30, 30), content="Text 1", metadata={}
        )
        element2 = PDFLayoutElement(
            element_id="elem2", element_type="text", page_number=1,
            bbox=(25, 25, 45, 45), content="Text 2", metadata={}
        )
        
        # Small overlap might not trigger default threshold
        assert element1.overlaps_with(element2, threshold=0.01)


class TestPDFPageAnalysis:
    """Test PDFPageAnalysis dataclass."""
    
    def test_page_analysis_creation(self):
        """Test PDFPageAnalysis creation."""
        analysis = PDFPageAnalysis(
            page_number=1,
            page_width=612.0,
            page_height=792.0,
            text_blocks=[],
            tables=[],
            images=[],
            layout_columns=2,
            reading_order=["elem1", "elem2"]
        )
        
        assert analysis.page_number == 1
        assert analysis.page_width == 612.0
        assert analysis.page_height == 792.0
        assert analysis.layout_columns == 2
        assert analysis.reading_order == ["elem1", "elem2"]


class TestAdvancedPDFProcessorInit:
    """Test AdvancedPDFProcessor initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default config."""
        with patch('src.advanced_pdf_processor.Config') as mock_config:
            mock_config.return_value = Mock()
            processor = AdvancedPDFProcessor()
            
            assert processor.config is not None
            assert isinstance(processor.processed_pages, list)
            assert isinstance(processor.extracted_elements, list)
            assert len(processor.processed_pages) == 0
            assert len(processor.extracted_elements) == 0
    
    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = Mock(spec=Config)
        processor = AdvancedPDFProcessor(config)
        
        assert processor.config == config
        assert isinstance(processor.processed_pages, list)
        assert isinstance(processor.extracted_elements, list)


class TestPDFProcessing:
    """Test main PDF processing functionality."""
    
    def test_process_pdf_file_not_found(self, pdf_processor):
        """Test processing non-existent PDF file."""
        documents, elements = pdf_processor.process_pdf("/nonexistent/file.pdf")
        
        assert documents == []
        assert elements == []
    
    @patch('pathlib.Path.exists', return_value=True)
    def test_process_pdf_success(self, mock_exists, pdf_processor):
        """Test successful PDF processing."""
        pdf_path = "/test/sample.pdf"
        
        with patch.object(pdf_processor, '_extract_text_with_layout') as mock_text, \
             patch.object(pdf_processor, '_extract_tables_from_pdf') as mock_tables, \
             patch.object(pdf_processor, '_extract_images_from_pdf') as mock_images, \
             patch.object(pdf_processor, '_analyze_page_layouts') as mock_layouts:
            
            # Mock return values
            mock_text.return_value = [Mock(spec=Document)]
            mock_tables.return_value = [Mock(spec=MultiModalElement)]
            mock_images.return_value = [Mock(spec=MultiModalElement)]
            
            documents, elements = pdf_processor.process_pdf(pdf_path)
            
            assert len(documents) == 1
            assert len(elements) == 2  # 1 table + 1 image
            
            mock_text.assert_called_once()
            mock_tables.assert_called_once()
            mock_images.assert_called_once()
            mock_layouts.assert_called_once()
    
    @patch('pathlib.Path.exists', return_value=True)
    def test_process_pdf_exception_handling(self, mock_exists, pdf_processor):
        """Test PDF processing exception handling."""
        pdf_path = "/test/sample.pdf"
        
        with patch.object(pdf_processor, '_extract_text_with_layout', side_effect=Exception("Text extraction failed")):
            documents, elements = pdf_processor.process_pdf(pdf_path)
            
            assert documents == []
            assert elements == []


class TestTextExtraction:
    """Test text extraction functionality."""
    
    @patch('src.advanced_pdf_processor.PDFPLUMBER_AVAILABLE', False)
    def test_extract_text_fallback_to_pypdf(self, pdf_processor):
        """Test text extraction fallback when pdfplumber not available."""
        pdf_path = Path("/test/sample.pdf")
        
        with patch('langchain_community.document_loaders.PyPDFLoader') as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [Mock(spec=Document)]
            mock_loader.return_value = mock_instance
            
            documents = pdf_processor._extract_text_with_layout(pdf_path)
            
            assert len(documents) == 1
            mock_loader.assert_called_once_with(str(pdf_path))
    
    @patch('src.advanced_pdf_processor.PDFPLUMBER_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.pdfplumber')
    def test_extract_text_with_pdfplumber(self, mock_pdfplumber, pdf_processor):
        """Test text extraction with pdfplumber."""
        pdf_path = Path("/test/sample.pdf")
        
        # Mock pdfplumber
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF content"
        mock_page.chars = [
            {'x0': 10, 'y0': 10, 'x1': 20, 'y1': 20, 'text': 'S'},
            {'x0': 20, 'y0': 10, 'x1': 30, 'y1': 20, 'text': 'a'}
        ]
        mock_page.width = 612
        mock_page.height = 792
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        with patch.object(pdf_processor, '_analyze_text_layout') as mock_layout:
            mock_layout.return_value = {"columns": 1, "layout_type": "single_column"}
            
            documents = pdf_processor._extract_text_with_layout(pdf_path)
            
            assert len(documents) == 1
            assert documents[0].page_content == "Sample PDF content"
            assert documents[0].metadata["page"] == 1
            assert documents[0].metadata["source"] == str(pdf_path)
    
    @patch('src.advanced_pdf_processor.PDFPLUMBER_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.pdfplumber')
    def test_extract_text_with_exception(self, mock_pdfplumber, pdf_processor):
        """Test text extraction with pdfplumber exception."""
        pdf_path = Path("/test/sample.pdf")
        
        mock_pdfplumber.open.side_effect = Exception("pdfplumber failed")
        
        with patch('langchain_community.document_loaders.PyPDFLoader') as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [Mock(spec=Document)]
            mock_loader.return_value = mock_instance
            
            documents = pdf_processor._extract_text_with_layout(pdf_path)
            
            assert len(documents) == 1
            mock_loader.assert_called_once()


class TestTextLayoutAnalysis:
    """Test text layout analysis functionality."""
    
    def test_analyze_text_layout_empty_chars(self, pdf_processor):
        """Test layout analysis with empty characters."""
        result = pdf_processor._analyze_text_layout([], 612, 792)
        
        assert result["columns"] == 1
        assert result["layout_type"] == "single_column"
    
    def test_analyze_text_layout_single_column(self, pdf_processor):
        """Test layout analysis for single column."""
        chars = [
            {'x0': 50, 'y0': 100, 'x1': 60, 'y1': 110},
            {'x0': 50, 'y0': 120, 'x1': 60, 'y1': 130},
            {'x0': 50, 'y0': 140, 'x1': 60, 'y1': 150}
        ]
        
        result = pdf_processor._analyze_text_layout(chars, 612, 792)
        
        assert isinstance(result, dict)
        assert "columns" in result
        assert "layout_type" in result
    
    def test_analyze_text_layout_exception_handling(self, pdf_processor):
        """Test layout analysis exception handling."""
        # Malformed character data
        chars = [{"invalid": "data"}]
        
        result = pdf_processor._analyze_text_layout(chars, 612, 792)
        
        assert result["columns"] == 1
        assert result["layout_type"] == "single_column"


class TestColumnDetection:
    """Test column detection functionality."""
    
    def test_detect_columns_empty_lines(self, pdf_processor):
        """Test column detection with empty lines."""
        result = pdf_processor._detect_columns({}, 612)
        
        assert result["num_columns"] == 1
        assert result["boundaries"] == []
        assert result["layout_type"] == "single_column"
    
    def test_detect_columns_single_column(self, pdf_processor):
        """Test detection of single column layout."""
        lines = {
            100.0: [{'x0': 50, 'x1': 200}],
            90.0: [{'x0': 50, 'x1': 180}],
            80.0: [{'x0': 50, 'x1': 220}]
        }
        
        result = pdf_processor._detect_columns(lines, 612)
        
        assert result["num_columns"] == 1
        assert result["layout_type"] == "single_column"
    
    def test_detect_columns_multi_column(self, pdf_processor):
        """Test detection of multi-column layout."""
        lines = {
            100.0: [{'x0': 50, 'x1': 200}, {'x0': 350, 'x1': 500}],
            90.0: [{'x0': 50, 'x1': 180}, {'x0': 350, 'x1': 480}],
            80.0: [{'x0': 50, 'x1': 220}, {'x0': 350, 'x1': 520}]
        }
        
        result = pdf_processor._detect_columns(lines, 612)
        
        # Should detect multiple start positions
        assert isinstance(result, dict)
        assert "num_columns" in result
        assert "layout_type" in result
    
    def test_detect_columns_exception_handling(self, pdf_processor):
        """Test column detection exception handling."""
        # Invalid line data
        lines = {100.0: [{"invalid": "data"}]}
        
        result = pdf_processor._detect_columns(lines, 612)
        
        assert result["num_columns"] == 1
        assert result["layout_type"] == "single_column"


class TestTableExtraction:
    """Test table extraction functionality."""
    
    @patch('src.advanced_pdf_processor.PDFPLUMBER_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.CAMELOT_AVAILABLE', False)
    @patch('src.advanced_pdf_processor.TABULA_AVAILABLE', False)
    def test_extract_tables_pdfplumber_only(self, pdf_processor):
        """Test table extraction with only pdfplumber available."""
        pdf_path = Path("/test/sample.pdf")
        
        with patch.object(pdf_processor, '_extract_tables_pdfplumber') as mock_pdfplumber, \
             patch.object(pdf_processor, '_deduplicate_tables') as mock_dedup:
            
            mock_pdfplumber.return_value = [Mock(spec=MultiModalElement)]
            mock_dedup.return_value = [Mock(spec=MultiModalElement)]
            
            tables = pdf_processor._extract_tables_from_pdf(pdf_path)
            
            assert len(tables) == 1
            mock_pdfplumber.assert_called_once_with(pdf_path)
            mock_dedup.assert_called_once()
    
    @patch('src.advanced_pdf_processor.PDFPLUMBER_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.pdfplumber')
    def test_extract_tables_pdfplumber_success(self, mock_pdfplumber, pdf_processor, sample_dataframe):
        """Test pdfplumber table extraction."""
        pdf_path = Path("/test/sample.pdf")
        
        # Mock table data
        table_data = [
            ['Name', 'Age', 'City'],
            ['Alice', '25', 'New York'],
            ['Bob', '30', 'London']
        ]
        
        mock_page = Mock()
        mock_page.extract_tables.return_value = [table_data]
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        with patch.object(pdf_processor, '_clean_table_dataframe') as mock_clean, \
             patch.object(pdf_processor, '_generate_table_description') as mock_desc:
            
            mock_clean.return_value = sample_dataframe
            mock_desc.return_value = "Sample table description"
            
            tables = pdf_processor._extract_tables_pdfplumber(pdf_path)
            
            assert len(tables) == 1
            assert tables[0].element_type == "table"
            assert tables[0].metadata["extraction_method"] == "pdfplumber"
    
    @patch('src.advanced_pdf_processor.CAMELOT_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.camelot')
    def test_extract_tables_camelot_success(self, mock_camelot, pdf_processor, sample_dataframe):
        """Test camelot table extraction."""
        pdf_path = Path("/test/sample.pdf")
        
        # Mock camelot table
        mock_table = Mock()
        mock_table.df = sample_dataframe.copy()
        mock_table.page = 1
        mock_table.accuracy = 85.5
        mock_table.whitespace = 12.3
        mock_table._bbox = (10, 10, 100, 100)
        
        mock_table_list = Mock()
        mock_table_list.__iter__ = Mock(return_value=iter([mock_table]))
        
        mock_camelot.read_pdf.return_value = mock_table_list
        
        with patch.object(pdf_processor, '_clean_table_dataframe') as mock_clean, \
             patch.object(pdf_processor, '_generate_table_description') as mock_desc:
            
            mock_clean.return_value = sample_dataframe
            mock_desc.return_value = "Sample table description"
            
            tables = pdf_processor._extract_tables_camelot(pdf_path)
            
            assert len(tables) == 1
            assert tables[0].element_type == "table"
            assert tables[0].metadata["extraction_method"] == "camelot"
            assert tables[0].confidence_score > 0
    
    @patch('src.advanced_pdf_processor.TABULA_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.tabula')
    def test_extract_tables_tabula_success(self, mock_tabula, pdf_processor, sample_dataframe):
        """Test tabula table extraction."""
        pdf_path = Path("/test/sample.pdf")
        
        mock_tabula.read_pdf.return_value = [sample_dataframe.copy()]
        
        with patch.object(pdf_processor, '_clean_table_dataframe') as mock_clean, \
             patch.object(pdf_processor, '_generate_table_description') as mock_desc:
            
            mock_clean.return_value = sample_dataframe
            mock_desc.return_value = "Sample table description"
            
            tables = pdf_processor._extract_tables_tabula(pdf_path)
            
            assert len(tables) == 1
            assert tables[0].element_type == "table"
            assert tables[0].metadata["extraction_method"] == "tabula"


class TestTableCleaning:
    """Test table cleaning functionality."""
    
    def test_clean_table_dataframe_basic(self, pdf_processor):
        """Test basic table cleaning."""
        # Create messy DataFrame
        df = pd.DataFrame({
            '  Name  ': ['Alice', 'Bob', None],
            'Age': [25, None, 35],
            '': ['', 'data', 'more'],  # Empty column name
            None: ['x', 'y', 'z']  # None column name
        })
        
        cleaned = pdf_processor._clean_table_dataframe(df)
        
        assert not cleaned.empty
        assert 'Name' in cleaned.columns or 'Name  ' in cleaned.columns
        # Should handle empty and None column names
    
    def test_clean_table_dataframe_duplicate_columns(self, pdf_processor):
        """Test cleaning DataFrame with duplicate column names."""
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob'],
            'Name': ['Smith', 'Jones'],  # Duplicate column
            'Age': [25, 30]
        })
        
        cleaned = pdf_processor._clean_table_dataframe(df)
        
        # Should handle duplicate columns
        assert not cleaned.empty
        assert len(cleaned.columns) == len(set(cleaned.columns))  # No duplicates
    
    def test_clean_table_dataframe_empty(self, pdf_processor):
        """Test cleaning empty DataFrame."""
        df = pd.DataFrame()
        
        cleaned = pdf_processor._clean_table_dataframe(df)
        
        assert cleaned.empty
    
    def test_clean_table_dataframe_all_na(self, pdf_processor):
        """Test cleaning DataFrame with all NA values."""
        df = pd.DataFrame({
            'A': [None, None, None],
            'B': [pd.NA, pd.NA, pd.NA]
        })
        
        cleaned = pdf_processor._clean_table_dataframe(df)
        
        # Should remove all-NA rows
        assert cleaned.empty or len(cleaned) < 3


class TestTableDescription:
    """Test table description generation."""
    
    def test_generate_table_description_basic(self, pdf_processor, sample_dataframe):
        """Test basic table description generation."""
        description = pdf_processor._generate_table_description(sample_dataframe, 1)
        
        assert isinstance(description, str)
        assert "3 rows" in description
        assert "3 columns" in description
        assert "page 1" in description
        assert "Name" in description
    
    def test_generate_table_description_unknown_page(self, pdf_processor, sample_dataframe):
        """Test table description with unknown page."""
        description = pdf_processor._generate_table_description(sample_dataframe, "unknown")
        
        assert isinstance(description, str)
        assert "page unknown" not in description
        assert "3 rows" in description
    
    def test_generate_table_description_empty_dataframe(self, pdf_processor):
        """Test description generation for empty DataFrame."""
        df = pd.DataFrame()
        description = pdf_processor._generate_table_description(df, 1)
        
        assert isinstance(description, str)
        assert "0 rows" in description
    
    def test_generate_table_description_exception(self, pdf_processor):
        """Test description generation exception handling."""
        # Pass invalid data to trigger exception
        description = pdf_processor._generate_table_description(None, 1)
        
        assert isinstance(description, str)
        # Should provide fallback description


class TestTableDeduplication:
    """Test table deduplication functionality."""
    
    def test_deduplicate_tables_empty_list(self, pdf_processor):
        """Test deduplication with empty table list."""
        result = pdf_processor._deduplicate_tables([])
        assert result == []
    
    def test_deduplicate_tables_single_table(self, pdf_processor, sample_dataframe):
        """Test deduplication with single table."""
        element = MultiModalElement(
            element_id="test_table",
            element_type="table",
            content=sample_dataframe,
            metadata={},
            text_description="Test table",
            confidence_score=0.8
        )
        
        result = pdf_processor._deduplicate_tables([element])
        assert len(result) == 1
    
    def test_tables_are_similar_identical(self, pdf_processor, sample_dataframe):
        """Test similarity detection for identical tables."""
        element1 = MultiModalElement(
            element_id="table1", element_type="table",
            content=sample_dataframe.copy(), metadata={},
            text_description="Table 1", confidence_score=0.8
        )
        element2 = MultiModalElement(
            element_id="table2", element_type="table",
            content=sample_dataframe.copy(), metadata={},
            text_description="Table 2", confidence_score=0.9
        )
        
        assert pdf_processor._tables_are_similar(element1, element2)
    
    def test_tables_are_similar_different(self, pdf_processor, sample_dataframe):
        """Test similarity detection for different tables."""
        df2 = pd.DataFrame({
            'Product': ['A', 'B', 'C'],
            'Price': [10, 20, 30]
        })
        
        element1 = MultiModalElement(
            element_id="table1", element_type="table",
            content=sample_dataframe, metadata={},
            text_description="Table 1", confidence_score=0.8
        )
        element2 = MultiModalElement(
            element_id="table2", element_type="table",
            content=df2, metadata={},
            text_description="Table 2", confidence_score=0.9
        )
        
        assert not pdf_processor._tables_are_similar(element1, element2)
    
    def test_tables_are_similar_non_dataframe(self, pdf_processor):
        """Test similarity check with non-DataFrame content."""
        element1 = MultiModalElement(
            element_id="elem1", element_type="table",
            content="not a dataframe", metadata={},
            text_description="Element 1", confidence_score=0.8
        )
        element2 = MultiModalElement(
            element_id="elem2", element_type="table",
            content="also not a dataframe", metadata={},
            text_description="Element 2", confidence_score=0.9
        )
        
        assert not pdf_processor._tables_are_similar(element1, element2)


class TestImageExtraction:
    """Test image extraction functionality."""
    
    @patch('src.advanced_pdf_processor.PYMUPDF_AVAILABLE', False)
    def test_extract_images_pymupdf_unavailable(self, pdf_processor):
        """Test image extraction when PyMuPDF is unavailable."""
        pdf_path = Path("/test/sample.pdf")
        
        images = pdf_processor._extract_images_from_pdf(pdf_path)
        
        assert images == []
    
    @patch('src.advanced_pdf_processor.PYMUPDF_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.fitz')
    def test_extract_images_success(self, mock_fitz, pdf_processor):
        """Test successful image extraction."""
        pdf_path = Path("/test/sample.pdf")
        
        # Mock PyMuPDF objects
        mock_pixmap = Mock()
        mock_pixmap.n = 3  # RGB
        mock_pixmap.alpha = 0
        mock_pixmap.tobytes.return_value = b"fake_image_data"
        
        mock_page = Mock()
        mock_page.get_images.return_value = [(123, 0, 100, 100, 0, "image", "name")]
        mock_page.get_image_rects.return_value = [(10, 10, 110, 110)]
        
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = Mock()
        
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Pixmap.return_value = mock_pixmap
        
        images = pdf_processor._extract_images_from_pdf(pdf_path)
        
        assert len(images) == 1
        assert images[0].element_type == "image"
        assert images[0].metadata["extraction_method"] == "pymupdf"
    
    @patch('src.advanced_pdf_processor.PYMUPDF_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.fitz')
    def test_extract_images_exception(self, mock_fitz, pdf_processor):
        """Test image extraction with exception."""
        pdf_path = Path("/test/sample.pdf")
        
        mock_fitz.open.side_effect = Exception("PyMuPDF error")
        
        images = pdf_processor._extract_images_from_pdf(pdf_path)
        
        assert images == []


class TestPageLayoutAnalysis:
    """Test page layout analysis functionality."""
    
    @patch('src.advanced_pdf_processor.PDFPLUMBER_AVAILABLE', False)
    def test_analyze_page_layouts_unavailable(self, pdf_processor):
        """Test layout analysis when pdfplumber is unavailable."""
        pdf_path = Path("/test/sample.pdf")
        
        # Should not crash
        pdf_processor._analyze_page_layouts(pdf_path)
        
        assert len(pdf_processor.processed_pages) == 0
    
    @patch('src.advanced_pdf_processor.PDFPLUMBER_AVAILABLE', True)
    @patch('src.advanced_pdf_processor.pdfplumber')
    def test_analyze_page_layouts_success(self, mock_pdfplumber, pdf_processor):
        """Test successful page layout analysis."""
        pdf_path = Path("/test/sample.pdf")
        
        mock_page = Mock()
        mock_page.width = 612
        mock_page.height = 792
        mock_page.chars = [
            {'x0': 50, 'y0': 100, 'x1': 60, 'y1': 110},
        ]
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        with patch.object(pdf_processor, '_analyze_text_layout') as mock_layout:
            mock_layout.return_value = {"columns": 1}
            
            pdf_processor._analyze_page_layouts(pdf_path)
            
            assert len(pdf_processor.processed_pages) == 1
            assert pdf_processor.processed_pages[0].page_number == 1


class TestProcessingSummary:
    """Test processing summary functionality."""
    
    def test_get_processing_summary_empty(self, pdf_processor):
        """Test processing summary with no processed data."""
        summary = pdf_processor.get_processing_summary()
        
        assert isinstance(summary, dict)
        assert summary["total_pages_analyzed"] == 0
        assert summary["total_elements_extracted"] == 0
        assert summary["elements_by_type"] == {}
        assert "capabilities" in summary
        assert "processing_methods_used" in summary
    
    def test_get_processing_summary_with_data(self, pdf_processor, sample_layout_element):
        """Test processing summary with processed data."""
        pdf_processor.processed_pages = [Mock(spec=PDFPageAnalysis)]
        pdf_processor.extracted_elements = [sample_layout_element]
        
        summary = pdf_processor.get_processing_summary()
        
        assert summary["total_pages_analyzed"] == 1
        assert summary["total_elements_extracted"] == 1
        assert "text" in summary["elements_by_type"]
        assert summary["elements_by_type"]["text"] == 1


class TestCacheClear:
    """Test cache clearing functionality."""
    
    def test_clear_cache(self, pdf_processor, sample_layout_element):
        """Test clearing processor cache."""
        pdf_processor.processed_pages = [Mock(spec=PDFPageAnalysis)]
        pdf_processor.extracted_elements = [sample_layout_element]
        
        pdf_processor.clear_cache()
        
        assert len(pdf_processor.processed_pages) == 0
        assert len(pdf_processor.extracted_elements) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_process_pdf_with_empty_path(self, pdf_processor):
        """Test processing with empty path."""
        documents, elements = pdf_processor.process_pdf("")
        
        assert documents == []
        assert elements == []
    
    def test_analyze_text_layout_with_malformed_chars(self, pdf_processor):
        """Test layout analysis with malformed character data."""
        malformed_chars = [
            {"missing_coordinates": True},
            {"x0": "not_a_number", "y0": 10},
        ]
        
        result = pdf_processor._analyze_text_layout(malformed_chars, 612, 792)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert "columns" in result
    
    def test_clean_table_with_extreme_values(self, pdf_processor):
        """Test table cleaning with extreme values."""
        df = pd.DataFrame({
            'A': ['normal', float('inf'), -float('inf')],
            'B': [1, 2, None],
            'C': ['', '   ', 'valid']
        })
        
        cleaned = pdf_processor._clean_table_dataframe(df)
        
        # Should handle extreme values gracefully
        assert isinstance(cleaned, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])