"""
Global pytest configuration and fixtures.
Shared fixtures and configurations for all test modules.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import pytest
from unittest.mock import Mock, patch

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Mock commonly problematic imports at the global level
MOCK_MODULES = [
    'chromadb',
    'chromadb.config',
    'pdfplumber', 
    'camelot',
    'fitz',
    'tabula',
    'nltk',
    'nltk.corpus',
    'nltk.tokenize',
    'sklearn',
    'sklearn.feature_extraction',
    'sklearn.feature_extraction.text',
    'sklearn.metrics',
    'sklearn.metrics.pairwise',
    'sklearn.cluster',
    'langchain_community.vectorstores',
    'langchain_community.embeddings',
    'langchain_community.document_loaders',
    'streamlit',
    'openai',
    'anthropic'
]

# Apply global mocking for problematic imports
for module_name in MOCK_MODULES:
    sys.modules[module_name] = Mock()


@pytest.fixture(scope="session")
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace for the test session."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    workspace = Path(temp_dir)
    
    # Create common directory structure
    (workspace / "uploads").mkdir()
    (workspace / "vector_store").mkdir() 
    (workspace / "cache").mkdir()
    (workspace / "logs").mkdir()
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for a single test."""
    temp_dir = tempfile.mkdtemp(prefix="test_")
    temp_path = Path(temp_dir)
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Mock configuration object with default values."""
    config = Mock()
    config.CHUNK_SIZE = 1000
    config.CHUNK_OVERLAP = 200
    config.MODEL_NAME = "gpt-3.5-turbo"
    config.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    config.TEMPERATURE = 0.7
    config.MAX_TOKENS = 1000
    config.VECTOR_STORE_TYPE = "chroma"
    config.VECTOR_STORE_DIR = "/tmp/vector_store"
    config.UPLOAD_DIR = "/tmp/uploads"
    config.SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md', '.xlsx', '.jpg', '.png']
    config.OPENAI_API_KEY = "test-openai-key"
    config.ANTHROPIC_API_KEY = None
    config.get_llm_provider.return_value = "openai"
    return config


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """
    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.
    
    Deep learning is part of a broader family of machine learning methods based on 
    artificial neural networks with representation learning.
    
    Natural language processing is a subfield of linguistics, computer science, and artificial
    intelligence concerned with the interactions between computers and human language.
    """


@pytest.fixture
def sample_pdf_content():
    """Sample PDF-like content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000015 00000 n \n0000000074 00000 n \n0000000131 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n207\n%%EOF"


@pytest.fixture
def create_temp_file():
    """Factory fixture to create temporary files."""
    created_files = []
    
    def _create_file(content: str, suffix: str = ".txt", mode: str = "w") -> Path:
        """Create a temporary file with given content."""
        temp_file = tempfile.NamedTemporaryFile(
            mode=mode, 
            suffix=suffix, 
            delete=False,
            encoding='utf-8' if 'b' not in mode else None
        )
        temp_file.write(content)
        temp_file.close()
        
        file_path = Path(temp_file.name)
        created_files.append(file_path)
        return file_path
    
    yield _create_file
    
    # Cleanup
    for file_path in created_files:
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing."""
    with patch('streamlit.session_state', {}), \
         patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.info') as mock_info, \
         patch('streamlit.file_uploader') as mock_uploader, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.slider') as mock_slider:
        
        # Configure mock behaviors
        mock_columns.return_value = [Mock(), Mock()]
        mock_uploader.return_value = None
        mock_button.return_value = False
        mock_selectbox.return_value = "Option 1"
        mock_text_input.return_value = ""
        mock_slider.return_value = 0.7
        
        yield {
            'sidebar': mock_sidebar,
            'columns': mock_columns,
            'write': mock_write,
            'error': mock_error,
            'success': mock_success,
            'warning': mock_warning,
            'info': mock_info,
            'file_uploader': mock_uploader,
            'button': mock_button,
            'selectbox': mock_selectbox,
            'text_input': mock_text_input,
            'slider': mock_slider
        }


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "pdf: mark test as requiring PDF processing")
    config.addinivalue_line("markers", "vectordb: mark test as requiring vector database")
    config.addinivalue_line("markers", "ai: mark test as requiring AI/LLM models")


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Add markers based on test location or name
    for item in items:
        # Mark all tests in unit directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark tests based on filename patterns
        if "pdf" in item.name.lower():
            item.add_marker(pytest.mark.pdf)
        if "vector" in item.name.lower():
            item.add_marker(pytest.mark.vectordb)
        if "ai" in item.name.lower() or "llm" in item.name.lower():
            item.add_marker(pytest.mark.ai)
        if "slow" in item.name.lower() or item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Setup for each test item."""
    # Skip slow tests unless explicitly requested
    if item.get_closest_marker("slow") and not item.config.getoption("--run-slow", default=False):
        pytest.skip("Slow test skipped (use --run-slow to include)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true", 
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False, 
        help="Run integration tests"
    )


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment before and after each test."""
    # Before test
    original_env = os.environ.copy()
    
    yield
    
    # After test - restore environment
    os.environ.clear()
    os.environ.update(original_env)


# Global test utilities
class TestHelpers:
    """Helper utilities for tests."""
    
    @staticmethod
    def create_mock_document(content: str = "Sample content", source: str = "test.txt", **metadata):
        """Create a mock document object."""
        from langchain.schema import Document
        return Document(
            page_content=content,
            metadata={"source": source, **metadata}
        )
    
    @staticmethod
    def create_mock_multimodal_element(
        element_id: str = "test_element",
        element_type: str = "text",
        content = "Test content",
        confidence: float = 0.9
    ):
        """Create a mock multimodal element."""
        from unittest.mock import Mock
        element = Mock()
        element.element_id = element_id
        element.element_type = element_type
        element.content = content
        element.confidence_score = confidence
        element.metadata = {}
        element.text_description = f"Mock {element_type} element"
        element.processing_method = "mock"
        return element


# Make helpers available globally
helpers = TestHelpers()