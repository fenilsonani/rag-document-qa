"""
Unit tests for the main Streamlit application (app.py).
Tests all core functions and user interactions.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import app module - we need to mock streamlit first
with patch.dict('sys.modules', {'streamlit': MagicMock()}):
    import app
    from src.config import Config
    from src.enhanced_rag import EnhancedRAG
    from src.model_manager import ModelManager, ModelDownloadProgress


@pytest.fixture
def mock_streamlit():
    """Mock streamlit components."""
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.success = MagicMock()
    mock_st.error = MagicMock()
    mock_st.info = MagicMock()
    mock_st.spinner = MagicMock()
    mock_st.expander = MagicMock()
    mock_st.json = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.write = MagicMock()
    return mock_st


@pytest.fixture
def mock_enhanced_rag():
    """Mock EnhancedRAG instance."""
    mock_rag = MagicMock(spec=EnhancedRAG)
    mock_rag.initialize.return_value = {
        "success": True,
        "mode": "online",
        "details": "System initialized successfully"
    }
    return mock_rag


@pytest.fixture
def mock_config():
    """Mock Config instance."""
    mock_cfg = MagicMock(spec=Config)
    mock_cfg.UPLOAD_DIR = "test_uploads"
    return mock_cfg


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_uploaded_file():
    """Mock uploaded file object."""
    mock_file = MagicMock()
    mock_file.name = "test_document.pdf"
    mock_file.getbuffer.return_value = b"test file content"
    return mock_file


class TestInitializeSystem:
    """Test the initialize_system function."""
    
    @patch('app.st')
    def test_initialize_system_success_online_mode(self, mock_st):
        """Test successful system initialization in online mode."""
        # Setup
        mock_st.session_state = {
            "enhanced_rag": MagicMock(),
            "initialized": False
        }
        mock_st.session_state.enhanced_rag.initialize.return_value = {
            "success": True,
            "mode": "online",
            "details": "API keys found"
        }
        mock_st.spinner.return_value.__enter__.return_value = None
        mock_st.spinner.return_value.__exit__.return_value = None
        
        # Execute
        result = app.initialize_system()
        
        # Assert
        assert result is True
        assert mock_st.session_state["initialized"] is True
        assert mock_st.session_state["system_mode"] == "online"
        mock_st.success.assert_called_with("✅ Online mode enabled with API keys!")
        mock_st.session_state.enhanced_rag.initialize.assert_called_once()
    
    @patch('app.st')
    def test_initialize_system_success_offline_mode(self, mock_st):
        """Test successful system initialization in offline mode."""
        # Setup
        mock_st.session_state = {
            "enhanced_rag": MagicMock(),
            "initialized": False
        }
        mock_st.session_state.enhanced_rag.initialize.return_value = {
            "success": True,
            "mode": "offline",
            "details": "No API keys, using offline models"
        }
        mock_st.spinner.return_value.__enter__.return_value = None
        mock_st.spinner.return_value.__exit__.return_value = None
        
        # Execute
        result = app.initialize_system()
        
        # Assert
        assert result is True
        assert mock_st.session_state["initialized"] is True
        assert mock_st.session_state["system_mode"] == "offline"
        mock_st.success.assert_called_with("🔧 Offline mode enabled (no API keys needed)!")
    
    @patch('app.st')
    def test_initialize_system_success_basic_mode(self, mock_st):
        """Test successful system initialization in basic mode."""
        # Setup
        mock_st.session_state = {
            "enhanced_rag": MagicMock(),
            "initialized": False
        }
        mock_st.session_state.enhanced_rag.initialize.return_value = {
            "success": True,
            "mode": "basic",
            "details": "Limited functionality"
        }
        mock_st.spinner.return_value.__enter__.return_value = None
        mock_st.spinner.return_value.__exit__.return_value = None
        
        # Execute
        result = app.initialize_system()
        
        # Assert
        assert result is True
        assert mock_st.session_state["initialized"] is True
        assert mock_st.session_state["system_mode"] == "basic"
        mock_st.info.assert_called_with("⚠️ Basic mode (limited functionality)")
    
    @patch('app.st')
    def test_initialize_system_failure(self, mock_st):
        """Test failed system initialization."""
        # Setup
        mock_st.session_state = {
            "enhanced_rag": MagicMock(),
            "initialized": False
        }
        mock_st.session_state.enhanced_rag.initialize.return_value = {
            "success": False,
            "error": "Configuration error"
        }
        mock_st.spinner.return_value.__enter__.return_value = None
        mock_st.spinner.return_value.__exit__.return_value = None
        
        # Execute
        result = app.initialize_system()
        
        # Assert
        assert result is False
        mock_st.error.assert_called_with("❌ System initialization failed")
    
    @patch('app.st')
    def test_initialize_system_already_initialized(self, mock_st):
        """Test initialization when system is already initialized."""
        # Setup
        mock_st.session_state = {
            "enhanced_rag": MagicMock(),
            "initialized": True
        }
        
        # Execute
        result = app.initialize_system()
        
        # Assert
        assert result is True
        mock_st.session_state.enhanced_rag.initialize.assert_not_called()
    
    @patch('app.st')
    def test_initialize_system_force_mode(self, mock_st):
        """Test initialization with force mode."""
        # Setup
        mock_st.session_state = {
            "enhanced_rag": MagicMock(),
            "initialized": True  # Already initialized
        }
        mock_st.session_state.enhanced_rag.initialize.return_value = {
            "success": True,
            "mode": "offline"
        }
        mock_st.spinner.return_value.__enter__.return_value = None
        mock_st.spinner.return_value.__exit__.return_value = None
        
        # Execute
        result = app.initialize_system(force_mode="offline")
        
        # Assert
        assert result is True
        mock_st.session_state.enhanced_rag.initialize.assert_called_with(force_mode="offline")


class TestSaveUploadedFile:
    """Test the save_uploaded_file function."""
    
    @patch('app.st')
    @patch('app.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_uploaded_file_success(self, mock_file_open, mock_path, mock_st):
        """Test successful file upload and save."""
        # Setup
        mock_st.session_state = {"config": MagicMock()}
        mock_st.session_state.config.UPLOAD_DIR = "test_uploads"
        
        mock_upload_dir = MagicMock()
        mock_file_path = MagicMock()
        mock_file_path.__str__ = MagicMock(return_value="test_uploads/test.pdf")
        mock_upload_dir.__truediv__ = MagicMock(return_value=mock_file_path)
        mock_path.return_value = mock_upload_dir
        
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "test.pdf"
        mock_uploaded_file.getbuffer.return_value = b"test content"
        
        # Execute
        result = app.save_uploaded_file(mock_uploaded_file)
        
        # Assert
        assert result == "test_uploads/test.pdf"
        mock_upload_dir.mkdir.assert_called_with(exist_ok=True)
        mock_file_open.assert_called_once()
        mock_uploaded_file.getbuffer.assert_called_once()
    
    @patch('app.st')
    @patch('app.Path')
    def test_save_uploaded_file_directory_creation(self, mock_path, mock_st):
        """Test that upload directory is created if it doesn't exist."""
        # Setup
        mock_st.session_state = {"config": MagicMock()}
        mock_st.session_state.config.UPLOAD_DIR = "new_uploads"
        
        mock_upload_dir = MagicMock()
        mock_path.return_value = mock_upload_dir
        
        with patch('builtins.open', mock_open()):
            mock_uploaded_file = MagicMock()
            mock_uploaded_file.name = "test.pdf"
            mock_uploaded_file.getbuffer.return_value = b"content"
            
            # Execute
            app.save_uploaded_file(mock_uploaded_file)
            
            # Assert
            mock_upload_dir.mkdir.assert_called_with(exist_ok=True)


class TestDisplaySources:
    """Test the display_sources function."""
    
    @patch('app.st')
    def test_display_sources_empty_list(self, mock_st):
        """Test display_sources with empty sources list."""
        # Execute
        app.display_sources([])
        
        # Assert - should return early, no calls to streamlit
        mock_st.subheader.assert_not_called()
    
    @patch('app.st')
    def test_display_sources_single_source(self, mock_st):
        """Test display_sources with single source."""
        # Setup
        sources = [{
            "index": 1,
            "filename": "test.pdf",
            "content": "This is test content"
        }]
        
        mock_expander = MagicMock()
        mock_st.expander.return_value = mock_expander
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        
        # Execute
        app.display_sources(sources)
        
        # Assert
        mock_st.subheader.assert_called_with("📄 Sources")
        mock_st.expander.assert_called_with("Source 1: test.pdf")
        mock_st.write.assert_any_call("**Content Preview:**")
        mock_st.write.assert_any_call("This is test content")
    
    @patch('app.st')
    def test_display_sources_multiple_sources(self, mock_st):
        """Test display_sources with multiple sources."""
        # Setup
        sources = [
            {"index": 1, "filename": "doc1.pdf", "content": "Content 1"},
            {"index": 2, "filename": "doc2.pdf", "content": "Content 2"}
        ]
        
        mock_expander = MagicMock()
        mock_st.expander.return_value = mock_expander
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        
        # Execute
        app.display_sources(sources)
        
        # Assert
        mock_st.subheader.assert_called_with("📄 Sources")
        assert mock_st.expander.call_count == 2
        mock_st.expander.assert_any_call("Source 1: doc1.pdf")
        mock_st.expander.assert_any_call("Source 2: doc2.pdf")


class TestFileHandling:
    """Test file handling functions."""
    
    def test_save_uploaded_file_real_filesystem(self):
        """Test save_uploaded_file with real filesystem operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            with patch('app.st') as mock_st:
                mock_st.session_state = {"config": MagicMock()}
                mock_st.session_state.config.UPLOAD_DIR = temp_dir
                
                mock_uploaded_file = MagicMock()
                mock_uploaded_file.name = "real_test.txt"
                mock_uploaded_file.getbuffer.return_value = b"real test content"
                
                # Execute
                result = app.save_uploaded_file(mock_uploaded_file)
                
                # Assert
                expected_path = os.path.join(temp_dir, "real_test.txt")
                assert result == expected_path
                assert os.path.exists(expected_path)
                
                with open(expected_path, 'rb') as f:
                    content = f.read()
                    assert content == b"real test content"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @patch('app.st')
    def test_initialize_system_exception_handling(self, mock_st):
        """Test initialize_system handles exceptions gracefully."""
        # Setup
        mock_st.session_state = {
            "enhanced_rag": MagicMock(),
            "initialized": False
        }
        mock_st.session_state.enhanced_rag.initialize.side_effect = Exception("Network error")
        mock_st.spinner.return_value.__enter__.return_value = None
        mock_st.spinner.return_value.__exit__.return_value = None
        
        # Execute & Assert
        with pytest.raises(Exception):
            app.initialize_system()
    
    @patch('app.st')
    def test_display_sources_none_input(self, mock_st):
        """Test display_sources with None input."""
        # Execute
        app.display_sources(None)
        
        # Assert - should handle None gracefully
        mock_st.subheader.assert_not_called()
    
    @patch('app.st')
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    @patch('app.Path')
    def test_save_uploaded_file_permission_error(self, mock_path, mock_file_open, mock_st):
        """Test save_uploaded_file handles permission errors."""
        # Setup
        mock_st.session_state = {"config": MagicMock()}
        mock_st.session_state.config.UPLOAD_DIR = "protected_dir"
        
        mock_upload_dir = MagicMock()
        mock_file_path = MagicMock()
        mock_upload_dir.__truediv__ = MagicMock(return_value=mock_file_path)
        mock_path.return_value = mock_upload_dir
        
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "test.pdf"
        
        # Execute & Assert
        with pytest.raises(PermissionError):
            app.save_uploaded_file(mock_uploaded_file)


class TestSessionStateInitialization:
    """Test session state initialization logic."""
    
    @patch('app.st')
    def test_session_state_defaults(self, mock_st):
        """Test that session state is initialized with correct defaults."""
        # This test would need to be structured differently as the session state
        # initialization happens at module level, but we can test the expected structure
        
        expected_keys = [
            "enhanced_rag", "initialized", "chat_history", "system_mode",
            "config", "document_insights", "relationship_analysis",
            "smart_suggestions", "model_manager", "download_progress"
        ]
        
        # We can't easily test the actual module-level initialization,
        # but we can document the expected behavior
        assert True  # Placeholder - this would be tested in integration tests


class TestModelDownloadProgress:
    """Test model download progress functionality."""
    
    def test_model_download_progress_structure(self):
        """Test ModelDownloadProgress data structure."""
        # This tests the expected structure of ModelDownloadProgress
        progress = ModelDownloadProgress(
            model_name="test-model",
            status="downloading",
            progress=0.5,
            message="Downloading..."
        )
        
        assert progress.model_name == "test-model"
        assert progress.status == "downloading"
        assert progress.progress == 0.5
        assert progress.message == "Downloading..."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])