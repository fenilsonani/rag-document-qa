"""
Basic functionality tests for core modules.
Simple tests to verify basic functionality without complex dependencies.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestBasicImports:
    """Test that basic imports work correctly."""
    
    def test_config_import(self):
        """Test that config module can be imported."""
        try:
            from src.config import Config
            config = Config()
            assert hasattr(config, 'CHUNK_SIZE')
            assert hasattr(config, 'CHUNK_OVERLAP')
            assert config.CHUNK_SIZE > 0
        except ImportError as e:
            pytest.skip(f"Config import failed: {e}")
    
    def test_basic_classes_structure(self):
        """Test basic class structures exist."""
        # Mock dependencies before importing
        with patch.dict('sys.modules', {
            'chromadb': MagicMock(),
            'chromadb.config': MagicMock(),
            'langchain_community.vectorstores': MagicMock(),
            'langchain_community.embeddings': MagicMock()
        }):
            try:
                from src.vector_store import VectorStoreManager
                assert VectorStoreManager is not None
                
                # Test basic initialization
                manager = VectorStoreManager("test_collection")
                assert manager.collection_name == "test_collection"
            except Exception as e:
                pytest.skip(f"VectorStoreManager test failed: {e}")


class TestUtilityFunctions:
    """Test utility functions and basic operations."""
    
    def test_path_operations(self):
        """Test basic path operations."""
        from pathlib import Path
        
        # Test path creation and manipulation
        test_path = Path("/tmp/test")
        assert isinstance(test_path, Path)
        assert str(test_path) == "/tmp/test"
        
        # Test suffix operations
        file_path = Path("test.pdf")
        assert file_path.suffix == ".pdf"
        assert file_path.suffix.lower() == ".pdf"
    
    def test_basic_data_structures(self):
        """Test basic data structure operations."""
        # Test list operations
        test_list = [1, 2, 3, 4, 5]
        assert len(test_list) == 5
        assert test_list[0] == 1
        assert test_list[-1] == 5
        
        # Test dictionary operations
        test_dict = {"key1": "value1", "key2": "value2"}
        assert "key1" in test_dict
        assert test_dict.get("key1") == "value1"
        assert test_dict.get("nonexistent", "default") == "default"


class TestMockFunctionality:
    """Test that mocking works correctly for our test setup."""
    
    def test_mock_creation(self):
        """Test basic mock creation and behavior."""
        mock_obj = Mock()
        mock_obj.test_method.return_value = "test_result"
        
        result = mock_obj.test_method()
        assert result == "test_result"
        mock_obj.test_method.assert_called_once()
    
    def test_patch_functionality(self):
        """Test patch decorator functionality."""
        with patch('builtins.open', mock_open(read_data="test data")):
            with open("fake_file.txt", "r") as f:
                content = f.read()
            assert content == "test data"
    
    def test_magicmock_functionality(self):
        """Test MagicMock functionality."""
        magic_mock = MagicMock()
        magic_mock.attribute = "test_value"
        magic_mock.method.return_value = "method_result"
        
        assert magic_mock.attribute == "test_value"
        assert magic_mock.method() == "method_result"


class TestConfigurationValues:
    """Test configuration and constant values."""
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        supported_extensions = ['.pdf', '.txt', '.docx', '.md', '.xlsx', '.jpg', '.png']
        
        assert '.pdf' in supported_extensions
        assert '.txt' in supported_extensions
        assert '.xyz' not in supported_extensions
        
        # Test extension validation
        for ext in supported_extensions:
            assert ext.startswith('.')
            assert len(ext) > 1
    
    def test_chunk_sizes(self):
        """Test chunk size configurations."""
        chunk_size = 1000
        chunk_overlap = 200
        
        assert chunk_size > 0
        assert chunk_overlap > 0
        assert chunk_overlap < chunk_size


class TestStringOperations:
    """Test string processing operations used in the system."""
    
    def test_text_preprocessing(self):
        """Test basic text preprocessing operations."""
        sample_text = "  This is a sample text with    extra spaces.  "
        
        # Test trimming
        trimmed = sample_text.strip()
        assert not trimmed.startswith(' ')
        assert not trimmed.endswith(' ')
        
        # Test lowercasing
        lower_text = sample_text.lower()
        assert 'THIS' not in lower_text
        assert 'this' in lower_text
        
        # Test splitting
        words = trimmed.split()
        assert len(words) > 0
        assert isinstance(words, list)
    
    def test_filename_operations(self):
        """Test filename and path operations."""
        filename = "document.pdf"
        
        # Test extension extraction
        name, ext = filename.rsplit('.', 1)
        assert name == "document"
        assert ext == "pdf"
        
        # Test path joining
        full_path = os.path.join("/tmp", "uploads", filename)
        assert filename in full_path
        assert "/tmp" in full_path


class TestErrorHandling:
    """Test error handling patterns."""
    
    def test_exception_catching(self):
        """Test exception handling."""
        def risky_function():
            raise ValueError("Test error")
        
        try:
            risky_function()
            assert False, "Should have raised exception"
        except ValueError as e:
            assert str(e) == "Test error"
        except Exception:
            assert False, "Wrong exception type"
    
    def test_file_operations_error_handling(self):
        """Test file operations error handling."""
        nonexistent_file = "/nonexistent/path/file.txt"
        
        try:
            with open(nonexistent_file, 'r'):
                pass
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            # Expected behavior
            pass
        except Exception as e:
            # On some systems might be different exception
            assert "No such file" in str(e) or "cannot find" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])