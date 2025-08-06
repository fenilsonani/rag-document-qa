"""
Unit tests for document_loader.py module.
Tests document loading, processing, splitting, and multimodal element extraction.
"""

import pytest
import sys
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock dependencies before importing
with patch.dict('sys.modules', {
    'langchain_community.document_loaders': MagicMock(),
    'advanced_pdf_processor': MagicMock(),
    'multimodal_rag': MagicMock(),
    'universal_file_processor': MagicMock()
}):
    from src.document_loader import DocumentProcessor
    from src.config import Config
    from langchain.schema import Document
    from src.multimodal_rag import MultiModalElement


@pytest.fixture
def temp_file():
    """Create temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("Sample document content for testing.")
        return tmp_file.name


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup is handled by system


@pytest.fixture
def mock_config():
    """Create mock Config instance."""
    config = Mock(spec=Config)
    config.SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md', '.xlsx', '.jpg', '.png']
    config.CHUNK_SIZE = 1000
    config.CHUNK_OVERLAP = 200
    config.UPLOAD_DIR = "/tmp/uploads"
    return config


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="First document content with machine learning concepts.",
            metadata={"source": "doc1.pdf", "page": 1}
        ),
        Document(
            page_content="Second document discussing artificial intelligence and neural networks.",
            metadata={"source": "doc2.pdf", "page": 2}
        ),
        Document(
            page_content="Third document about natural language processing applications.",
            metadata={"source": "doc3.pdf", "page": 1}
        )
    ]


@pytest.fixture
def sample_multimodal_elements():
    """Create sample multimodal elements for testing."""
    return [
        MultiModalElement(
            element_id="table_1",
            element_type="table",
            content={"data": "table"},
            metadata={"source": "doc1.pdf"},
            text_description="Sample table",
            confidence_score=0.9,
            processing_method="pdfplumber"
        ),
        MultiModalElement(
            element_id="image_1",
            element_type="image",
            content=b"fake_image_data",
            metadata={"source": "doc2.pdf"},
            text_description="Sample image",
            confidence_score=0.8,
            processing_method="pymupdf"
        )
    ]


@pytest.fixture
def document_processor(mock_config):
    """Create DocumentProcessor instance with mocked dependencies."""
    with patch('src.document_loader.Config', return_value=mock_config), \
         patch('src.document_loader.RecursiveCharacterTextSplitter') as mock_splitter, \
         patch('src.document_loader.AdvancedPDFProcessor') as mock_pdf_proc, \
         patch('src.document_loader.UniversalFileProcessor') as mock_universal:
        
        mock_splitter.return_value = Mock()
        mock_pdf_proc.return_value = Mock()
        mock_universal.return_value = Mock()
        
        processor = DocumentProcessor()
        return processor


class TestDocumentProcessorInit:
    """Test DocumentProcessor initialization."""
    
    def test_initialization_success(self, mock_config):
        """Test successful DocumentProcessor initialization."""
        with patch('src.document_loader.Config', return_value=mock_config), \
             patch('src.document_loader.RecursiveCharacterTextSplitter') as mock_splitter, \
             patch('src.document_loader.AdvancedPDFProcessor'), \
             patch('src.document_loader.UniversalFileProcessor'):
            
            mock_text_splitter = Mock()
            mock_splitter.return_value = mock_text_splitter
            
            processor = DocumentProcessor()
            
            assert processor.config == mock_config
            assert processor.text_splitter == mock_text_splitter
            assert hasattr(processor, 'advanced_pdf_processor')
            assert hasattr(processor, 'universal_processor')
            assert isinstance(processor.multimodal_elements, list)
            assert len(processor.multimodal_elements) == 0
    
    def test_text_splitter_configuration(self, mock_config):
        """Test text splitter is configured correctly."""
        with patch('src.document_loader.Config', return_value=mock_config), \
             patch('src.document_loader.RecursiveCharacterTextSplitter') as mock_splitter, \
             patch('src.document_loader.AdvancedPDFProcessor'), \
             patch('src.document_loader.UniversalFileProcessor'):
            
            DocumentProcessor()
            
            mock_splitter.assert_called_once_with(
                chunk_size=mock_config.CHUNK_SIZE,
                chunk_overlap=mock_config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )


class TestLoadDocument:
    """Test single document loading functionality."""
    
    def test_load_document_unsupported_extension(self, document_processor):
        """Test loading document with unsupported extension."""
        with pytest.raises(ValueError, match="Unsupported file type: .xyz"):
            document_processor.load_document("/path/to/file.xyz")
    
    def test_load_document_pdf_success(self, document_processor, sample_documents, sample_multimodal_elements):
        """Test successful PDF document loading."""
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value.suffix.lower.return_value = ".pdf"
            
            document_processor.advanced_pdf_processor.process_pdf.return_value = (
                sample_documents, sample_multimodal_elements
            )
            
            result = document_processor.load_document("test.pdf")
            
            assert result == sample_documents
            assert len(document_processor.multimodal_elements) == len(sample_multimodal_elements)
            document_processor.advanced_pdf_processor.process_pdf.assert_called_once_with("test.pdf")
    
    def test_load_document_pdf_fallback(self, document_processor):
        """Test PDF loading fallback to PyPDFLoader."""
        with patch('pathlib.Path') as mock_path, \
             patch('src.document_loader.PyPDFLoader') as mock_loader:
            
            mock_path.return_value.suffix.lower.return_value = ".pdf"
            document_processor.advanced_pdf_processor.process_pdf.return_value = ([], [])
            
            mock_pdf_loader = Mock()
            mock_pdf_loader.load.return_value = [Mock(spec=Document)]
            mock_loader.return_value = mock_pdf_loader
            
            result = document_processor.load_document("test.pdf")
            
            assert len(result) == 1
            mock_loader.assert_called_once_with("test.pdf")
    
    def test_load_document_txt_success(self, document_processor):
        """Test successful text document loading."""
        with patch('pathlib.Path') as mock_path, \
             patch('src.document_loader.TextLoader') as mock_loader:
            
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".txt"
            mock_path_instance.name = "test.txt"
            mock_path.return_value = mock_path_instance
            
            mock_text_loader = Mock()
            mock_document = Document(page_content="Text content", metadata={})
            mock_text_loader.load.return_value = [mock_document]
            mock_loader.return_value = mock_text_loader
            
            result = document_processor.load_document("test.txt")
            
            assert len(result) == 1
            assert result[0].metadata["source"] == "test.txt"
            assert result[0].metadata["filename"] == "test.txt"
            assert result[0].metadata["file_type"] == ".txt"
            mock_loader.assert_called_once_with("test.txt", encoding="utf-8")
    
    def test_load_document_docx_success(self, document_processor):
        """Test successful DOCX document loading."""
        with patch('pathlib.Path') as mock_path, \
             patch('src.document_loader.Docx2txtLoader') as mock_loader:
            
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".docx"
            mock_path_instance.name = "test.docx"
            mock_path.return_value = mock_path_instance
            
            mock_docx_loader = Mock()
            mock_document = Document(page_content="DOCX content", metadata={})
            mock_docx_loader.load.return_value = [mock_document]
            mock_loader.return_value = mock_docx_loader
            
            result = document_processor.load_document("test.docx")
            
            assert len(result) == 1
            assert result[0].metadata["file_type"] == ".docx"
            mock_loader.assert_called_once_with("test.docx")
    
    def test_load_document_markdown_success(self, document_processor):
        """Test successful Markdown document loading."""
        with patch('pathlib.Path') as mock_path, \
             patch('src.document_loader.UnstructuredMarkdownLoader') as mock_loader:
            
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".md"
            mock_path_instance.name = "test.md"
            mock_path.return_value = mock_path_instance
            
            mock_md_loader = Mock()
            mock_document = Document(page_content="# Markdown content", metadata={})
            mock_md_loader.load.return_value = [mock_document]
            mock_loader.return_value = mock_md_loader
            
            result = document_processor.load_document("test.md")
            
            assert len(result) == 1
            assert result[0].metadata["file_type"] == ".md"
            mock_loader.assert_called_once_with("test.md")
    
    def test_load_document_universal_processor_success(self, document_processor, sample_documents, sample_multimodal_elements):
        """Test successful loading with universal processor."""
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".xlsx"
            mock_path_instance.name = "test.xlsx"
            mock_path.return_value = mock_path_instance
            
            # Mock successful universal processing
            mock_result = Mock()
            mock_result.success = True
            mock_result.documents = sample_documents
            mock_result.multimodal_elements = sample_multimodal_elements
            document_processor.universal_processor.process_file.return_value = mock_result
            
            result = document_processor.load_document("test.xlsx")
            
            assert result == sample_documents
            assert len(document_processor.multimodal_elements) == len(sample_multimodal_elements)
    
    def test_load_document_universal_processor_fallback(self, document_processor):
        """Test universal processor fallback to text reading."""
        with patch('pathlib.Path') as mock_path, \
             patch('builtins.open', mock_open(read_data="Fallback text content")):
            
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".xlsx"
            mock_path_instance.name = "test.xlsx"
            mock_path.return_value = mock_path_instance
            
            # Mock failed universal processing
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "Processing failed"
            document_processor.universal_processor.process_file.return_value = mock_result
            
            result = document_processor.load_document("test.xlsx")
            
            assert len(result) == 1
            assert result[0].page_content == "Fallback text content"
            assert result[0].metadata["processing_method"] == "text_fallback"
    
    def test_load_document_complete_failure(self, document_processor):
        """Test document loading complete failure."""
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".xlsx"
            mock_path.return_value = mock_path_instance
            
            # Mock failed universal processing
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "Processing failed"
            document_processor.universal_processor.process_file.return_value = mock_result
            
            # Mock failed text fallback
            with patch('builtins.open', side_effect=Exception("File read failed")):
                with pytest.raises(ValueError, match="Could not process .xlsx file"):
                    document_processor.load_document("test.xlsx")
    
    def test_load_document_exception_handling(self, document_processor):
        """Test document loading exception handling."""
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value.suffix.lower.return_value = ".txt"
            
            # Mock loader to raise exception
            with patch('src.document_loader.TextLoader', side_effect=Exception("Loader failed")):
                with pytest.raises(RuntimeError, match="Error loading document"):
                    document_processor.load_document("test.txt")


class TestLoadMultipleDocuments:
    """Test multiple document loading functionality."""
    
    def test_load_multiple_documents_success(self, document_processor, sample_documents):
        """Test successful loading of multiple documents."""
        file_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
        
        with patch.object(document_processor, 'load_document') as mock_load:
            mock_load.side_effect = [
                [sample_documents[0]],
                [sample_documents[1]],
                [sample_documents[2]]
            ]
            
            result = document_processor.load_multiple_documents(file_paths)
            
            assert len(result) == 3
            assert mock_load.call_count == 3
    
    def test_load_multiple_documents_partial_failure(self, document_processor, sample_documents):
        """Test loading multiple documents with some failures."""
        file_paths = ["doc1.txt", "bad_doc.txt", "doc3.txt"]
        
        with patch.object(document_processor, 'load_document') as mock_load:
            mock_load.side_effect = [
                [sample_documents[0]],
                Exception("Load failed"),
                [sample_documents[2]]
            ]
            
            result = document_processor.load_multiple_documents(file_paths)
            
            assert len(result) == 2  # Only successful loads
            assert mock_load.call_count == 3
    
    def test_load_multiple_documents_empty_list(self, document_processor):
        """Test loading empty list of documents."""
        result = document_processor.load_multiple_documents([])
        
        assert result == []


class TestSplitDocuments:
    """Test document splitting functionality."""
    
    def test_split_documents_success(self, document_processor, sample_documents):
        """Test successful document splitting."""
        mock_chunks = [
            Document(page_content="Chunk 1", metadata={"source": "doc1.pdf"}),
            Document(page_content="Chunk 2", metadata={"source": "doc1.pdf"}),
            Document(page_content="Chunk 3", metadata={"source": "doc2.pdf"})
        ]
        
        document_processor.text_splitter.split_documents.return_value = mock_chunks
        
        result = document_processor.split_documents(sample_documents)
        
        assert len(result) == 3
        # Check that chunk metadata was added
        for i, chunk in enumerate(result):
            assert chunk.metadata["chunk_id"] == i
            assert "chunk_size" in chunk.metadata
        
        document_processor.text_splitter.split_documents.assert_called_once_with(sample_documents)
    
    def test_split_documents_empty_list(self, document_processor):
        """Test splitting empty document list."""
        result = document_processor.split_documents([])
        
        assert result == []
        document_processor.text_splitter.split_documents.assert_not_called()


class TestProcessDocuments:
    """Test complete document processing pipeline."""
    
    def test_process_documents_success(self, document_processor, sample_documents):
        """Test successful document processing pipeline."""
        file_paths = ["doc1.pdf", "doc2.txt"]
        mock_chunks = [Mock(spec=Document), Mock(spec=Document)]
        
        with patch.object(document_processor, 'load_multiple_documents') as mock_load, \
             patch.object(document_processor, 'split_documents') as mock_split:
            
            mock_load.return_value = sample_documents
            mock_split.return_value = mock_chunks
            
            result = document_processor.process_documents(file_paths)
            
            assert result == mock_chunks
            mock_load.assert_called_once_with(file_paths)
            mock_split.assert_called_once_with(sample_documents)
    
    def test_process_documents_no_successful_loads(self, document_processor):
        """Test processing when no documents are successfully loaded."""
        file_paths = ["bad_doc.pdf"]
        
        with patch.object(document_processor, 'load_multiple_documents') as mock_load:
            mock_load.return_value = []
            
            with pytest.raises(ValueError, match="No documents were successfully loaded"):
                document_processor.process_documents(file_paths)


class TestFileManagement:
    """Test file management functionality."""
    
    def test_get_uploaded_files_success(self, document_processor, temp_dir, mock_config):
        """Test getting uploaded files from directory."""
        mock_config.SUPPORTED_EXTENSIONS = ['.pdf', '.txt']
        
        # Create test files
        test_files = [
            Path(temp_dir) / "doc1.pdf",
            Path(temp_dir) / "doc2.txt", 
            Path(temp_dir) / "doc3.xyz",  # Unsupported
            Path(temp_dir) / "subdir"     # Directory
        ]
        
        for file_path in test_files[:2]:
            file_path.touch()
        test_files[2].touch()
        test_files[3].mkdir()
        
        with patch.object(document_processor.config, 'UPLOAD_DIR', temp_dir):
            result = document_processor.get_uploaded_files()
            
            # Should only return supported file types
            assert len(result) == 2
            assert any("doc1.pdf" in path for path in result)
            assert any("doc2.txt" in path for path in result)
            assert not any("doc3.xyz" in path for path in result)
    
    def test_get_uploaded_files_nonexistent_directory(self, document_processor):
        """Test getting uploaded files from non-existent directory."""
        with patch.object(document_processor.config, 'UPLOAD_DIR', "/nonexistent/path"):
            result = document_processor.get_uploaded_files()
            
            assert result == []
    
    def test_get_uploaded_files_custom_directory(self, document_processor, temp_dir):
        """Test getting uploaded files from custom directory."""
        test_file = Path(temp_dir) / "custom.pdf"
        test_file.touch()
        
        result = document_processor.get_uploaded_files(temp_dir)
        
        assert len(result) == 1
        assert "custom.pdf" in result[0]
    
    def test_validate_file_success(self, document_processor, temp_file):
        """Test successful file validation."""
        # Create a small file with supported extension
        Path(temp_file).write_text("test content")
        
        result = document_processor.validate_file(temp_file)
        
        assert result is True
    
    def test_validate_file_not_exists(self, document_processor):
        """Test file validation for non-existent file."""
        result = document_processor.validate_file("/nonexistent/file.txt")
        
        assert result is False
    
    def test_validate_file_unsupported_extension(self, document_processor, temp_dir):
        """Test file validation for unsupported extension."""
        unsupported_file = Path(temp_dir) / "test.xyz"
        unsupported_file.touch()
        
        result = document_processor.validate_file(str(unsupported_file))
        
        assert result is False
    
    def test_validate_file_too_large(self, document_processor, temp_dir):
        """Test file validation for files too large."""
        large_file = Path(temp_dir) / "large.txt"
        large_file.touch()
        
        # Mock file size to be > 50MB
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 51 * 1024 * 1024  # 51MB
            
            result = document_processor.validate_file(str(large_file))
            
            assert result is False


class TestMultimodalElements:
    """Test multimodal element management."""
    
    def test_get_multimodal_elements(self, document_processor, sample_multimodal_elements):
        """Test getting all multimodal elements."""
        document_processor.multimodal_elements = sample_multimodal_elements
        
        result = document_processor.get_multimodal_elements()
        
        assert result == sample_multimodal_elements
    
    def test_get_tables(self, document_processor, sample_multimodal_elements):
        """Test getting only table elements."""
        document_processor.multimodal_elements = sample_multimodal_elements
        
        result = document_processor.get_tables()
        
        table_elements = [elem for elem in result if elem.element_type == "table"]
        assert len(table_elements) == 1
        assert table_elements[0].element_id == "table_1"
    
    def test_get_images(self, document_processor, sample_multimodal_elements):
        """Test getting only image elements."""
        document_processor.multimodal_elements = sample_multimodal_elements
        
        result = document_processor.get_images()
        
        image_elements = [elem for elem in result if elem.element_type in ["image", "chart"]]
        assert len(image_elements) == 1
        assert image_elements[0].element_id == "image_1"
    
    def test_get_processing_summary(self, document_processor, sample_multimodal_elements):
        """Test getting processing summary."""
        document_processor.multimodal_elements = sample_multimodal_elements
        
        # Mock PDF processor summary
        pdf_summary = {"test": "summary"}
        document_processor.advanced_pdf_processor.get_processing_summary.return_value = pdf_summary
        
        result = document_processor.get_processing_summary()
        
        assert result["total_multimodal_elements"] == 2
        assert result["tables"] == 1
        assert result["images"] == 1
        assert "pdfplumber" in result["extraction_methods"]
        assert "pymupdf" in result["extraction_methods"]
        assert result["avg_confidence"] == 0.85  # (0.9 + 0.8) / 2
        assert result["pdf_processing"] == pdf_summary
    
    def test_get_processing_summary_empty(self, document_processor):
        """Test processing summary with no multimodal elements."""
        result = document_processor.get_processing_summary()
        
        assert result["total_multimodal_elements"] == 0
        assert result["tables"] == 0
        assert result["images"] == 0
        assert result["avg_confidence"] == 0
    
    def test_clear_multimodal_cache(self, document_processor, sample_multimodal_elements):
        """Test clearing multimodal cache."""
        document_processor.multimodal_elements = sample_multimodal_elements.copy()
        
        document_processor.clear_multimodal_cache()
        
        assert len(document_processor.multimodal_elements) == 0
        document_processor.advanced_pdf_processor.clear_cache.assert_called_once()


class TestSupportedFormats:
    """Test supported formats functionality."""
    
    def test_get_supported_formats(self, document_processor, mock_config):
        """Test getting supported formats information."""
        mock_config.SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx']
        
        format_details = {"categories": "test"}
        document_processor.universal_processor.get_supported_formats.return_value = format_details
        
        result = document_processor.get_supported_formats()
        
        assert result["total_supported_extensions"] == 3
        assert result["supported_extensions"] == ['.pdf', '.txt', '.docx']
        assert result["format_categories"] == format_details
    
    def test_validate_file_support_pdf(self, document_processor):
        """Test file support validation for PDF files."""
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".pdf"
            mock_path_instance.name = "test.pdf"
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            result = document_processor.validate_file_support("test.pdf")
            
            assert result["is_supported"] is True
            assert result["file_exists"] is True
            assert result["processing_method"] == "advanced_pdf_processor"
            assert "text_extraction" in result["available_features"]
            assert "table_extraction" in result["available_features"]
            assert result["estimated_processing_time"] == "30-120 seconds"
    
    def test_validate_file_support_excel(self, document_processor):
        """Test file support validation for Excel files."""
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".xlsx"
            mock_path_instance.name = "test.xlsx"
            mock_path_instance.exists.return_value = True
            mock_path_instance.stat.return_value.st_size = 1024 * 1024  # 1MB
            mock_path.return_value = mock_path_instance
            
            result = document_processor.validate_file_support("test.xlsx")
            
            assert result["processing_method"] == "universal_processor_excel"
            assert "multi_sheet_extraction" in result["available_features"]
            assert result["file_size_mb"] == 1.0
    
    def test_validate_file_support_unsupported(self, document_processor):
        """Test file support validation for unsupported files."""
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".xyz"
            mock_path_instance.name = "test.xyz"
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            result = document_processor.validate_file_support("test.xyz")
            
            assert result["is_supported"] is False
            assert "error" in result
            assert "Unsupported file type" in result["error"]
    
    def test_validate_file_support_missing_file(self, document_processor):
        """Test file support validation for missing files."""
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".txt"
            mock_path_instance.name = "missing.txt"
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance
            
            result = document_processor.validate_file_support("missing.txt")
            
            assert result["file_exists"] is False
            assert "error" in result
            assert result["error"] == "File not found"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_load_document_empty_path(self, document_processor):
        """Test loading document with empty path."""
        with pytest.raises(RuntimeError):
            document_processor.load_document("")
    
    def test_multimodal_elements_persistence(self, document_processor, sample_multimodal_elements):
        """Test that multimodal elements persist across operations."""
        # Add elements from first operation
        document_processor.multimodal_elements.extend(sample_multimodal_elements[:1])
        
        # Add elements from second operation
        document_processor.multimodal_elements.extend(sample_multimodal_elements[1:])
        
        assert len(document_processor.multimodal_elements) == 2
    
    def test_unicode_filename_handling(self, document_processor):
        """Test handling of unicode filenames."""
        unicode_path = "测试文档.txt"
        
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".txt"
            mock_path_instance.name = unicode_path
            mock_path.return_value = mock_path_instance
            
            with patch('src.document_loader.TextLoader') as mock_loader:
                mock_text_loader = Mock()
                mock_document = Document(page_content="Unicode content", metadata={})
                mock_text_loader.load.return_value = [mock_document]
                mock_loader.return_value = mock_text_loader
                
                result = document_processor.load_document(unicode_path)
                
                assert len(result) == 1
                assert result[0].metadata["filename"] == unicode_path
    
    def test_large_document_handling(self, document_processor):
        """Test handling of very large documents."""
        large_content = "A" * 10000  # 10KB content
        
        with patch('pathlib.Path') as mock_path, \
             patch('src.document_loader.TextLoader') as mock_loader:
            
            mock_path_instance = Mock()
            mock_path_instance.suffix.lower.return_value = ".txt"
            mock_path_instance.name = "large.txt"
            mock_path.return_value = mock_path_instance
            
            mock_text_loader = Mock()
            mock_document = Document(page_content=large_content, metadata={})
            mock_text_loader.load.return_value = [mock_document]
            mock_loader.return_value = mock_text_loader
            
            result = document_processor.load_document("large.txt")
            
            assert len(result) == 1
            assert len(result[0].page_content) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])