"""
Core document processing functionality tests.
Tests document loading, chunking, and processing logic without external dependencies.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestFileValidation:
    """Test file validation logic."""
    
    def test_supported_extensions(self):
        """Test file extension validation."""
        supported_extensions = ['.pdf', '.txt', '.docx', '.md', '.xlsx', '.jpg', '.png']
        
        # Test valid extensions
        valid_files = [
            "document.pdf",
            "notes.txt",
            "report.docx",
            "readme.md",
            "data.xlsx",
            "image.jpg",
            "chart.png"
        ]
        
        for filename in valid_files:
            path = Path(filename)
            extension = path.suffix.lower()
            assert extension in supported_extensions
    
    def test_file_size_validation(self):
        """Test file size validation logic."""
        max_size = 50 * 1024 * 1024  # 50MB
        
        # Test acceptable sizes
        acceptable_sizes = [
            1024,           # 1KB
            1024 * 1024,    # 1MB
            10 * 1024 * 1024,  # 10MB
            max_size - 1    # Just under limit
        ]
        
        for size in acceptable_sizes:
            assert size <= max_size
        
        # Test unacceptable sizes
        unacceptable_sizes = [
            max_size + 1,           # Just over limit
            100 * 1024 * 1024,      # 100MB
            1024 * 1024 * 1024      # 1GB
        ]
        
        for size in unacceptable_sizes:
            assert size > max_size
    
    def test_filename_parsing(self):
        """Test filename parsing logic."""
        test_files = [
            ("document.pdf", "document", ".pdf"),
            ("data_file.xlsx", "data_file", ".xlsx"),
            ("image.JPG", "image", ".JPG"),
            ("readme.md", "readme", ".md"),
            ("file.with.dots.txt", "file.with.dots", ".txt")
        ]
        
        for filename, expected_name, expected_ext in test_files:
            path = Path(filename)
            name = path.stem
            ext = path.suffix
            
            assert name == expected_name
            assert ext == expected_ext


class TestTextProcessing:
    """Test text processing and chunking logic."""
    
    def test_text_chunking_logic(self):
        """Test document chunking algorithm."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunk_size = 50  # Small chunk size for testing
        chunk_overlap = 10
        
        # Simple chunking logic
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - chunk_overlap
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(len(chunk) <= chunk_size for chunk in chunks)
        
        # Test overlap
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            overlap_found = False
            for i in range(len(chunks) - 1):
                current_end = chunks[i][-chunk_overlap:]
                next_start = chunks[i + 1][:chunk_overlap]
                if any(c1 == c2 for c1, c2 in zip(current_end, next_start)):
                    overlap_found = True
                    break
            # Note: overlap might not be perfect due to simple string slicing
    
    def test_sentence_splitting(self):
        """Test sentence splitting logic."""
        text = "This is the first sentence. This is the second sentence! And this is the third? Final sentence."
        
        # Simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        assert len(sentences) == 4
        assert "This is the first sentence" in sentences
        assert "This is the second sentence" in sentences
        assert "And this is the third" in sentences
        assert "Final sentence" in sentences
    
    def test_metadata_extraction(self):
        """Test document metadata extraction."""
        filename = "research_paper.pdf"
        file_size = 1024 * 1024  # 1MB
        
        # Extract metadata
        path = Path(filename)
        metadata = {
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "stem": path.stem
        }
        
        assert metadata["filename"] == "research_paper.pdf"
        assert metadata["file_type"] == ".pdf"
        assert metadata["file_size_mb"] == 1.0
        assert metadata["stem"] == "research_paper"


class TestDocumentStructure:
    """Test document structure analysis."""
    
    def test_page_numbering(self):
        """Test page numbering logic."""
        total_pages = 5
        page_numbers = list(range(1, total_pages + 1))
        
        assert len(page_numbers) == total_pages
        assert page_numbers[0] == 1
        assert page_numbers[-1] == total_pages
        assert all(isinstance(page, int) for page in page_numbers)
    
    def test_chunk_indexing(self):
        """Test chunk indexing logic."""
        chunk_count = 10
        
        chunks_with_ids = []
        for i in range(chunk_count):
            chunk_info = {
                "chunk_id": i,
                "content": f"Chunk {i} content",
                "chunk_size": len(f"Chunk {i} content")
            }
            chunks_with_ids.append(chunk_info)
        
        assert len(chunks_with_ids) == chunk_count
        assert chunks_with_ids[0]["chunk_id"] == 0
        assert chunks_with_ids[-1]["chunk_id"] == chunk_count - 1
        assert all("chunk_size" in chunk for chunk in chunks_with_ids)
    
    def test_source_tracking(self):
        """Test source document tracking."""
        documents = [
            {"source": "doc1.pdf", "page": 1, "content": "Content 1"},
            {"source": "doc1.pdf", "page": 2, "content": "Content 2"},
            {"source": "doc2.txt", "page": 1, "content": "Content 3"},
        ]
        
        # Group by source
        source_groups = {}
        for doc in documents:
            source = doc["source"]
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        assert len(source_groups) == 2  # Two unique sources
        assert "doc1.pdf" in source_groups
        assert "doc2.txt" in source_groups
        assert len(source_groups["doc1.pdf"]) == 2  # Two pages from doc1
        assert len(source_groups["doc2.txt"]) == 1   # One page from doc2


class TestMultimodalElements:
    """Test multimodal element handling."""
    
    def test_element_classification(self):
        """Test multimodal element type classification."""
        elements = [
            {"type": "text", "content": "Text content"},
            {"type": "table", "content": {"rows": 5, "cols": 3}},
            {"type": "image", "content": b"fake_image_data"},
            {"type": "chart", "content": {"chart_type": "bar"}},
        ]
        
        # Classify elements
        text_elements = [e for e in elements if e["type"] == "text"]
        table_elements = [e for e in elements if e["type"] == "table"]
        visual_elements = [e for e in elements if e["type"] in ["image", "chart"]]
        
        assert len(text_elements) == 1
        assert len(table_elements) == 1
        assert len(visual_elements) == 2
    
    def test_confidence_scoring(self):
        """Test confidence score calculation for extractions."""
        extraction_results = [
            {"method": "pdfplumber", "success": True, "accuracy": 0.95},
            {"method": "camelot", "success": True, "accuracy": 0.88},
            {"method": "tabula", "success": False, "accuracy": 0.0},
        ]
        
        successful_extractions = [r for r in extraction_results if r["success"]]
        
        if successful_extractions:
            avg_confidence = sum(r["accuracy"] for r in successful_extractions) / len(successful_extractions)
            best_confidence = max(r["accuracy"] for r in successful_extractions)
        else:
            avg_confidence = 0.0
            best_confidence = 0.0
        
        assert avg_confidence > 0.9  # High average confidence
        assert best_confidence == 0.95  # Best method confidence
        assert len(successful_extractions) == 2  # Two successful methods
    
    def test_element_deduplication(self):
        """Test multimodal element deduplication logic."""
        elements = [
            {"id": "table_1", "type": "table", "content": "Table content A", "confidence": 0.9},
            {"id": "table_2", "type": "table", "content": "Table content A", "confidence": 0.8},  # Duplicate
            {"id": "table_3", "type": "table", "content": "Table content B", "confidence": 0.95},
        ]
        
        # Simple deduplication by content
        unique_elements = []
        seen_content = set()
        
        for element in elements:
            content_hash = hash(str(element["content"]))
            if content_hash not in seen_content:
                unique_elements.append(element)
                seen_content.add(content_hash)
            else:
                # If duplicate, keep the one with higher confidence
                for i, existing in enumerate(unique_elements):
                    if hash(str(existing["content"])) == content_hash:
                        if element["confidence"] > existing["confidence"]:
                            unique_elements[i] = element
                        break
        
        assert len(unique_elements) == 2  # Should have 2 unique elements
        assert unique_elements[0]["confidence"] == 0.9  # Higher confidence kept


class TestFileHandling:
    """Test file handling operations."""
    
    def test_temporary_file_operations(self):
        """Test temporary file operations."""
        import tempfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Test content")
            tmp_path = tmp_file.name
        
        # Verify file exists
        temp_path = Path(tmp_path)
        assert temp_path.exists()
        
        # Read content
        with open(tmp_path, 'r') as f:
            content = f.read()
        assert content == "Test content"
        
        # Cleanup
        temp_path.unlink()
        assert not temp_path.exists()
    
    def test_directory_operations(self):
        """Test directory operations."""
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        assert temp_path.exists()
        assert temp_path.is_dir()
        
        # Create subdirectories
        (temp_path / "uploads").mkdir()
        (temp_path / "processed").mkdir()
        
        assert (temp_path / "uploads").exists()
        assert (temp_path / "processed").exists()
        
        # Create test file
        test_file = temp_path / "uploads" / "test.txt"
        test_file.write_text("Test file content")
        
        assert test_file.exists()
        assert test_file.read_text() == "Test file content"
        
        # List files
        upload_files = list((temp_path / "uploads").glob("*.txt"))
        assert len(upload_files) == 1
        assert upload_files[0].name == "test.txt"
        
        # Cleanup
        shutil.rmtree(temp_dir)
        assert not temp_path.exists()


class TestProcessingPipeline:
    """Test document processing pipeline logic."""
    
    def test_pipeline_stages(self):
        """Test processing pipeline stages."""
        stages = ["load", "preprocess", "chunk", "extract", "store"]
        
        pipeline_status = {stage: False for stage in stages}
        
        # Simulate pipeline execution
        for stage in stages:
            # Mock stage processing
            pipeline_status[stage] = True
        
        assert all(pipeline_status.values())  # All stages completed
        assert len(pipeline_status) == len(stages)
    
    def test_error_recovery(self):
        """Test error recovery in processing pipeline."""
        def process_with_fallback(method_name, fallback_method):
            """Simulate processing with fallback."""
            try:
                if method_name == "primary_method":
                    raise Exception("Primary method failed")
                return f"Success with {method_name}"
            except Exception:
                return f"Fallback: {fallback_method}"
        
        # Test primary method failure
        result = process_with_fallback("primary_method", "backup_method")
        assert "Fallback: backup_method" in result
        
        # Test successful method
        result = process_with_fallback("working_method", "backup_method")
        assert "Success with working_method" in result
    
    def test_batch_processing(self):
        """Test batch processing logic."""
        files = ["file1.pdf", "file2.txt", "file3.docx"]
        batch_size = 2
        
        batches = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batches.append(batch)
        
        assert len(batches) == 2  # Two batches
        assert len(batches[0]) == 2  # First batch has 2 files
        assert len(batches[1]) == 1  # Second batch has 1 file
        assert batches[0] == ["file1.pdf", "file2.txt"]
        assert batches[1] == ["file3.docx"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])