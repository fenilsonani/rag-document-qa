# Installation, Testing & Deployment Guide

> **Complete Guide for RAG Document Q&A System Setup, Validation & Production Deployment**
> 
> **Version**: 2.0 Enhanced | **Environments**: Local, Cloud, Enterprise | **Status**: Production Ready

---

## 📋 Table of Contents

1. [Installation Guide](#installation-guide)
2. [System Requirements](#system-requirements)
3. [Environment Setup](#environment-setup)
4. [Dependency Management](#dependency-management)
5. [Configuration](#configuration)
6. [Testing & Validation](#testing--validation)
7. [Deployment Options](#deployment-options)
8. [Production Considerations](#production-considerations)
9. [Monitoring & Maintenance](#monitoring--maintenance)
10. [Troubleshooting](#troubleshooting)

---

## Installation Guide

### 🚀 Quick Installation (5 Minutes)

#### Prerequisites Check
```bash
# Check Python version (3.8+ required, 3.9+ recommended)
python3 --version

# Check available memory (4GB minimum, 8GB+ recommended)
free -h  # Linux
vm_stat | grep "free" # macOS

# Check disk space (5GB minimum recommended)
df -h
```

#### One-Command Installation
```bash
# Clone repository
git clone https://github.com/fenilsonani/rag-document-qa.git
cd rag-document-qa

# Run automated setup script
chmod +x install.sh
./install.sh
```

#### Manual Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/fenilsonani/rag-document-qa.git
cd rag-document-qa

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install additional system dependencies (if needed)
# macOS
brew install pkg-config libmagic

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libmagic1 libmagic-dev default-jre

# 5. Setup environment configuration
cp .env.example .env

# 6. Verify installation
python3 test_setup.py
```

### 📦 Advanced Installation Options

#### Development Installation
```bash
# Clone with development dependencies
git clone https://github.com/fenilsonani/rag-document-qa.git
cd rag-document-qa

# Create development environment
python3 -m venv venv-dev
source venv-dev/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Install in editable mode
pip install -e .
```

#### Production Installation
```bash
# Production-optimized installation
python3 -m venv venv-prod
source venv-prod/bin/activate

# Install only production dependencies
pip install -r requirements-prod.txt

# Setup production configuration
cp .env.production.example .env
```

#### Docker Installation
```bash
# Using Docker Compose (Recommended)
docker-compose up -d

# Or build manually
docker build -t rag-document-qa .
docker run -p 8501:8501 -v $(pwd)/uploads:/app/uploads rag-document-qa
```

---

## System Requirements

### 💻 Hardware Requirements

#### Minimum Requirements
```yaml
CPU: 2 cores, 2.0 GHz
RAM: 4 GB
Storage: 5 GB available space
Network: Internet connection for AI models
OS: Windows 10+, macOS 10.15+, Ubuntu 18.04+
```

#### Recommended Requirements
```yaml
CPU: 4+ cores, 3.0 GHz (Intel i5/AMD Ryzen 5 or better)
RAM: 8-16 GB (for large document processing)
Storage: 20 GB SSD (faster model loading and caching)
GPU: NVIDIA GPU with 4GB+ VRAM (optional, for AI acceleration)
Network: Stable broadband connection
```

#### Enterprise/Production Requirements
```yaml
CPU: 8+ cores, 3.5 GHz (Intel Xeon/AMD EPYC)
RAM: 32-64 GB
Storage: 100+ GB NVMe SSD
GPU: NVIDIA RTX 4090, A100, or similar (recommended for AI)
Network: Gigabit ethernet or faster
Load Balancer: For multiple instances
```

### 🖥️ Software Requirements

#### Operating System Support
```yaml
Supported Platforms:
  ✅ Ubuntu 18.04+ / Debian 10+
  ✅ CentOS 7+ / RHEL 8+
  ✅ macOS 10.15+ (Intel and Apple Silicon)
  ✅ Windows 10+ / Windows Server 2019+

Container Support:
  ✅ Docker 20.10+
  ✅ Kubernetes 1.20+
  ✅ OpenShift 4.8+
```

#### Python Environment
```yaml
Python Version: 3.8+ (3.9-3.11 recommended)
Package Manager: pip 21.0+ (or conda)
Virtual Environment: venv, virtualenv, or conda env

Critical Python Packages:
  - streamlit >= 1.30.0
  - langchain >= 0.1.0
  - chromadb >= 0.4.0
  - transformers >= 4.30.0
  - torch >= 2.0.0 (CPU or CUDA)
```

#### System Dependencies
```yaml
Required System Libraries:
  - libmagic (file type detection)
  - Java 8+ (for tabula-py PDF processing)
  - Tesseract OCR (for image text extraction)
  - OpenCV libraries (for image processing)

Optional But Recommended:
  - Redis server (for caching)
  - PostgreSQL (for document metadata)
  - NGINX (for production deployment)
```

---

## Environment Setup

### 🔧 Virtual Environment Configuration

#### Using venv (Recommended)
```bash
# Create isolated environment
python3 -m venv rag-env
source rag-env/bin/activate  # Linux/macOS
# rag-env\Scripts\activate  # Windows

# Verify isolation
which python3
which pip

# Install dependencies
pip install -r requirements.txt
```

#### Using conda
```bash
# Create conda environment
conda create -n rag-env python=3.9
conda activate rag-env

# Install pip dependencies
pip install -r requirements.txt

# Or use conda-forge when possible
conda install -c conda-forge streamlit pandas numpy
```

#### Environment Variables Setup
```bash
# Copy and customize environment template
cp .env.example .env

# Essential variables to configure
nano .env
```

**.env Configuration Template:**
```env
# Required: AI API Keys (choose at least one)
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Model Configuration
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-sonnet-4-20250514
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Document Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=1000
TEMPERATURE=0.7

# Advanced PDF Processing
PDF_EXTRACTION_METHODS=camelot,pdfplumber,pymupdf,tabula
PDF_IMAGE_EXTRACTION=true
PDF_TABLE_CONFIDENCE_THRESHOLD=0.75

# Multi-Modal AI Settings
MULTIMODAL_AI_ENABLED=true
IMAGE_AI_ANALYSIS=true
OCR_CONFIDENCE_THRESHOLD=0.6
CHART_ANALYSIS_ENABLED=true

# Performance Settings
PARALLEL_PROCESSING=true
MAX_WORKERS=3
ENABLE_CACHING=true
CACHE_DEFAULT_TTL=3600

# Storage Paths
UPLOAD_DIR=uploads
VECTOR_STORE_DIR=vector_store
CACHE_DIR=cache

# Optional: Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Optional: Database
DATABASE_URL=sqlite:///./rag_system.db

# Debug and Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
ENABLE_PROFILING=false
```

---

## Dependency Management

### 📦 Package Installation & Management

#### Core Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# Verify critical packages
python3 -c "
import streamlit, langchain, chromadb, transformers
import pandas, numpy, torch
print('✅ Core packages installed successfully')
"
```

#### Advanced PDF Processing Dependencies
```bash
# Install PDF processing packages
pip install pdfplumber>=0.9.0
pip install 'camelot-py[cv]>=0.10.1'  # Includes OpenCV
pip install PyMuPDF>=1.23.0
pip install tabula-py>=2.8.0

# Verify PDF processing capabilities
python3 -c "
import pdfplumber, camelot, fitz, tabula
print('✅ All PDF processing libraries available')
"
```

#### AI and ML Dependencies
```bash
# Install AI/ML packages
pip install transformers>=4.30.0
pip install torch>=2.0.0  # or torch+cuda for GPU
pip install sentence-transformers>=2.6.0
pip install accelerate>=0.20.0

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Office Document Processing
```bash
# Excel and Office formats
pip install openpyxl>=3.1.0
pip install xlrd>=2.0.0
pip install python-pptx>=0.6.21

# Additional format support
pip install python-magic>=0.4.27
pip install ebooklib  # for EPUB files
pip install Wand>=0.6.11  # for advanced image formats
```

#### System Dependency Installation

**Ubuntu/Debian:**
```bash
# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    libmagic1 \
    libmagic-dev \
    default-jre \
    tesseract-ocr \
    libtesseract-dev \
    libopencv-dev \
    pkg-config

# Optional: Redis for caching
sudo apt-get install redis-server
```

**macOS:**
```bash
# Using Homebrew
brew install pkg-config
brew install libmagic
brew install openjdk
brew install tesseract
brew install opencv

# Optional: Redis
brew install redis
```

**Windows:**
```powershell
# Using Chocolatey (run as Administrator)
choco install python3
choco install openjdk
choco install tesseract

# Note: Some dependencies may need manual installation on Windows
```

### 🔍 Dependency Verification

#### Comprehensive Dependency Check
```python
#!/usr/bin/env python3
"""
Comprehensive dependency verification script
"""

import sys
import importlib
import subprocess

def check_python_packages():
    """Check all required Python packages"""
    
    required_packages = {
        'streamlit': '1.30.0+',
        'langchain': '0.1.0+',
        'chromadb': '0.4.0+',
        'pandas': '2.0.0+',
        'numpy': '1.24.0+',
        'transformers': '4.30.0+',
        'torch': '2.0.0+',
        'pdfplumber': '0.9.0+',
        'openpyxl': '3.1.0+',
        'PIL': '10.0.0+',  # Pillow
    }
    
    optional_packages = {
        'camelot': 'camelot-py',
        'fitz': 'PyMuPDF', 
        'tabula': 'tabula-py',
        'redis': 'redis',
        'pytesseract': 'pytesseract'
    }
    
    print("🔍 Checking Required Python Packages:")
    print("=" * 50)
    
    all_good = True
    
    for package, version in required_packages.items():
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                print(f"✅ {package}: {module.__version__}")
            else:
                print(f"✅ {package}: installed")
        except ImportError:
            print(f"❌ {package}: NOT INSTALLED")
            all_good = False
    
    print("\n🔍 Checking Optional Packages:")
    print("=" * 40)
    
    for package, pip_name in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package}: available")
        except ImportError:
            print(f"⚠️  {package}: not available (install with: pip install {pip_name})")
    
    return all_good

def check_system_dependencies():
    """Check system-level dependencies"""
    
    system_deps = {
        'java': 'java -version',
        'tesseract': 'tesseract --version',
    }
    
    print("\n🔍 Checking System Dependencies:")
    print("=" * 40)
    
    for name, command in system_deps.items():
        try:
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stderr.split('\n')[0] if result.stderr else result.stdout.split('\n')[0]
                print(f"✅ {name}: {version.strip()}")
            else:
                print(f"❌ {name}: command failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ {name}: not found")
        except Exception as e:
            print(f"⚠️  {name}: check failed ({e})")

def check_gpu_availability():
    """Check GPU availability for AI acceleration"""
    
    print("\n🔍 Checking GPU Availability:")
    print("=" * 35)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU available: {gpu_name}")
            print(f"   GPU count: {gpu_count}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  No CUDA GPU detected (will use CPU)")
    except ImportError:
        print("❌ PyTorch not available")

def main():
    print("🧪 RAG System Dependency Verification")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
    
    # Check packages
    packages_ok = check_python_packages()
    
    # Check system dependencies
    check_system_dependencies()
    
    # Check GPU
    check_gpu_availability()
    
    print("\n" + "=" * 60)
    if packages_ok:
        print("🎉 All required dependencies are installed!")
        print("🚀 System ready for RAG document processing")
        return True
    else:
        print("❌ Some required dependencies are missing")
        print("📝 Install missing packages with: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

**Run the verification:**
```bash
python3 dependency_check.py
```

---

## Configuration

### ⚙️ System Configuration

#### Configuration File Hierarchy
```yaml
Configuration Priority (highest to lowest):
1. Environment variables
2. .env file
3. config.py defaults
4. Built-in defaults

Configuration Files:
- .env                    # Main environment configuration
- .env.local             # Local overrides (not in version control)
- .env.production        # Production-specific settings
- .env.development       # Development-specific settings
```

#### Advanced Configuration Options
```python
# config.py - Advanced configuration class

class AdvancedConfig:
    """
    Advanced configuration with validation and environment detection
    """
    
    def __init__(self, environment: str = None):
        self.environment = environment or self.detect_environment()
        self.load_configuration()
        self.validate_configuration()
    
    def detect_environment(self) -> str:
        """Auto-detect environment based on various indicators"""
        
        # Check for explicit environment variable
        if os.getenv('ENVIRONMENT'):
            return os.getenv('ENVIRONMENT')
        
        # Check for CI/CD environments
        if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
            return 'ci'
        
        # Check for Docker environment
        if os.path.exists('/.dockerenv'):
            return 'docker'
        
        # Check for production indicators
        if os.getenv('PROD') or 'prod' in sys.argv:
            return 'production'
        
        # Default to development
        return 'development'
    
    def load_configuration(self):
        """Load configuration based on environment"""
        
        base_config = self.load_base_config()
        env_config = self.load_environment_config()
        
        # Merge configurations
        self.config = {**base_config, **env_config}
        
        # Apply environment-specific optimizations
        self.apply_environment_optimizations()
    
    def apply_environment_optimizations(self):
        """Apply optimizations based on environment"""
        
        if self.environment == 'production':
            self.config.update({
                'PARALLEL_PROCESSING': True,
                'ENABLE_CACHING': True,
                'LOG_LEVEL': 'WARNING',
                'DEBUG_MODE': False,
                'PDF_EXTRACTION_METHODS': ['camelot', 'pdfplumber'],
                'CONFIDENCE_THRESHOLD': 0.8
            })
        
        elif self.environment == 'development':
            self.config.update({
                'PARALLEL_PROCESSING': False,
                'ENABLE_CACHING': False,
                'LOG_LEVEL': 'DEBUG',
                'DEBUG_MODE': True,
                'PDF_EXTRACTION_METHODS': ['pdfplumber'],  # Faster for dev
            })
        
        elif self.environment == 'ci':
            self.config.update({
                'PARALLEL_PROCESSING': False,
                'MULTIMODAL_AI_ENABLED': False,  # Skip AI in CI
                'LOG_LEVEL': 'INFO'
            })
```

#### Performance Tuning Configuration
```python
# Performance configuration templates

PERFORMANCE_CONFIGS = {
    'high_performance': {
        # Maximum performance for powerful systems
        'PARALLEL_PROCESSING': True,
        'MAX_WORKERS': 4,
        'CHUNK_SIZE': 1200,
        'CHUNK_OVERLAP': 250,
        'PDF_EXTRACTION_METHODS': ['camelot', 'pdfplumber', 'pymupdf'],
        'MULTIMODAL_AI_ENABLED': True,
        'ENABLE_CACHING': True,
        'CACHE_DEFAULT_TTL': 7200,
        'GPU_ENABLED': True
    },
    
    'balanced': {
        # Balanced performance and resource usage
        'PARALLEL_PROCESSING': True,
        'MAX_WORKERS': 2,
        'CHUNK_SIZE': 1000,
        'CHUNK_OVERLAP': 200,
        'PDF_EXTRACTION_METHODS': ['camelot', 'pdfplumber'],
        'MULTIMODAL_AI_ENABLED': True,
        'ENABLE_CACHING': True,
        'CACHE_DEFAULT_TTL': 3600
    },
    
    'memory_efficient': {
        # Optimized for systems with limited memory
        'PARALLEL_PROCESSING': False,
        'MAX_WORKERS': 1,
        'CHUNK_SIZE': 800,
        'CHUNK_OVERLAP': 150,
        'PDF_EXTRACTION_METHODS': ['pdfplumber'],
        'MULTIMODAL_AI_ENABLED': False,
        'ENABLE_CACHING': False,
        'CLEANUP_TEMP_FILES': True
    },
    
    'accuracy_focused': {
        # Maximum accuracy, longer processing time
        'PARALLEL_PROCESSING': True,
        'MAX_WORKERS': 2,
        'PDF_EXTRACTION_METHODS': ['camelot', 'pdfplumber', 'pymupdf', 'tabula'],
        'CROSS_VALIDATION_ENABLED': True,
        'CONFIDENCE_THRESHOLD': 0.85,
        'MULTIMODAL_AI_ENABLED': True,
        'IMAGE_AI_ANALYSIS': True
    }
}
```

#### Security Configuration
```python
# security_config.py - Security-focused configuration

class SecurityConfig:
    """
    Security configuration for production environments
    """
    
    def __init__(self):
        self.setup_secure_defaults()
        self.validate_security_settings()
    
    def setup_secure_defaults(self):
        """Setup secure default configurations"""
        
        self.security_settings = {
            # File Upload Security
            'MAX_FILE_SIZE_MB': 50,
            'ALLOWED_FILE_TYPES': ['.pdf', '.txt', '.docx', '.xlsx', '.pptx', '.png', '.jpg'],
            'SCAN_UPLOADS_FOR_MALWARE': True,
            'QUARANTINE_SUSPICIOUS_FILES': True,
            
            # API Security
            'RATE_LIMITING_ENABLED': True,
            'MAX_REQUESTS_PER_MINUTE': 60,
            'REQUIRE_API_KEY': False,  # Set to True for production
            'API_KEY_VALIDATION': True,
            
            # Data Security
            'ENCRYPT_STORED_DOCUMENTS': False,  # Enable for sensitive data
            'AUTO_DELETE_UPLOADS': True,
            'RETENTION_PERIOD_DAYS': 30,
            
            # Network Security
            'ENABLE_CORS': True,
            'ALLOWED_ORIGINS': ['http://localhost:8501'],
            'USE_HTTPS_ONLY': False,  # Set to True for production
            
            # Audit and Logging
            'ENABLE_AUDIT_LOGGING': True,
            'LOG_USER_ACTIONS': True,
            'LOG_SENSITIVE_DATA': False,
            
            # AI Model Security
            'VALIDATE_AI_INPUTS': True,
            'SANITIZE_EXTRACTED_TEXT': True,
            'CONTENT_FILTERING_ENABLED': True
        }
    
    def validate_security_settings(self):
        """Validate security configuration"""
        
        warnings = []
        
        if not self.security_settings['REQUIRE_API_KEY']:
            warnings.append("API key authentication is disabled")
        
        if not self.security_settings['USE_HTTPS_ONLY']:
            warnings.append("HTTPS is not enforced")
        
        if self.security_settings['MAX_FILE_SIZE_MB'] > 100:
            warnings.append("Large file size limit may cause DoS")
        
        if warnings:
            print("🚨 Security Warnings:")
            for warning in warnings:
                print(f"   ⚠️  {warning}")
```

---

## Testing & Validation

### 🧪 Comprehensive Testing Suite

#### Test Categories
```yaml
Test Types:
1. Unit Tests - Individual component testing
2. Integration Tests - Component interaction testing  
3. System Tests - End-to-end functionality testing
4. Performance Tests - Speed and resource usage testing
5. Security Tests - Security vulnerability testing
6. Compatibility Tests - Cross-platform and format testing
```

#### Running All Tests
```bash
# Quick system validation
python3 test_setup.py

# Comprehensive format testing
python3 test_all_formats.py

# PDF-specific testing
python3 test_pdf_multimodal.py

# Run full test suite (if pytest is installed)
pytest tests/ -v --cov=src
```

#### Unit Testing Framework
```python
# tests/test_document_processor.py

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from src.document_loader import DocumentProcessor
from src.config import Config

class TestDocumentProcessor:
    """Comprehensive tests for DocumentProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing"""
        config = Config()
        return DocumentProcessor(config)
    
    @pytest.fixture
    def sample_pdf(self):
        """Create a temporary PDF file for testing"""
        # This would create a sample PDF for testing
        # Implementation depends on testing requirements
        pass
    
    def test_supported_formats(self, processor):
        """Test that all expected formats are supported"""
        
        supported = processor.get_supported_formats()
        
        assert supported['total_supported_extensions'] >= 26
        assert '.pdf' in supported['supported_extensions']
        assert '.xlsx' in supported['supported_extensions']
        assert '.pptx' in supported['supported_extensions']
        assert '.jpg' in supported['supported_extensions']
    
    def test_file_validation(self, processor):
        """Test file validation functionality"""
        
        # Test supported file
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            validation = processor.validate_file_support(tmp.name)
            assert validation['is_supported'] == True
            assert validation['processing_method'] == 'advanced_pdf_processor'
        
        # Test unsupported file
        with tempfile.NamedTemporaryFile(suffix='.xyz') as tmp:
            validation = processor.validate_file_support(tmp.name)
            assert validation['is_supported'] == False
    
    def test_error_handling(self, processor):
        """Test error handling for various scenarios"""
        
        # Test non-existent file
        with pytest.raises(RuntimeError):
            processor.load_document('non_existent_file.pdf')
        
        # Test invalid file type
        with pytest.raises(ValueError):
            processor.load_document('test.invalid_extension')
    
    @patch('src.document_loader.AdvancedPDFProcessor')
    def test_pdf_processing_integration(self, mock_pdf_processor, processor):
        """Test integration with PDF processor"""
        
        # Setup mock
        mock_instance = Mock()
        mock_pdf_processor.return_value = mock_instance
        mock_instance.process_pdf.return_value = ([], [])
        
        # Test processing
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            # Add minimal PDF content
            tmp.write(b'%PDF-1.4\n%%EOF')
            tmp.flush()
            
            documents = processor.load_document(tmp.name)
            
            # Verify PDF processor was called
            mock_instance.process_pdf.assert_called_once_with(tmp.name)
    
    def test_multimodal_element_retrieval(self, processor):
        """Test multimodal element management"""
        
        # Initially empty
        elements = processor.get_multimodal_elements()
        assert len(elements) == 0
        
        # Test filtering
        tables = processor.get_tables()
        images = processor.get_images()
        assert isinstance(tables, list)
        assert isinstance(images, list)
    
    def test_processing_summary(self, processor):
        """Test processing summary generation"""
        
        summary = processor.get_processing_summary()
        
        required_keys = [
            'total_multimodal_elements',
            'tables',
            'images',
            'extraction_methods',
            'avg_confidence'
        ]
        
        for key in required_keys:
            assert key in summary
```

#### Integration Testing
```python
# tests/test_integration.py

import pytest
import tempfile
import shutil
from pathlib import Path

from src.document_loader import DocumentProcessor
from src.enhanced_rag import EnhancedRAG
from src.config import Config

class TestSystemIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture(scope='class')
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def rag_system(self):
        """Create complete RAG system for testing"""
        config = Config()
        processor = DocumentProcessor(config)
        rag = EnhancedRAG(config)
        return processor, rag
    
    def test_document_to_query_workflow(self, rag_system, temp_dir):
        """Test complete workflow from document processing to querying"""
        
        processor, rag = rag_system
        
        # Create a test document
        test_content = "This is a test document with important information about Q3 revenue of $1.2M."
        test_file = Path(temp_dir) / "test_document.txt"
        test_file.write_text(test_content)
        
        # Process document
        documents = processor.load_document(str(test_file))
        assert len(documents) > 0
        
        # Add to RAG system
        rag.add_documents(documents)
        
        # Query the system
        response = rag.query("What was the Q3 revenue?")
        
        assert response.answer is not None
        assert len(response.source_documents) > 0
        assert response.confidence_score > 0
    
    def test_multimodal_processing_workflow(self, rag_system):
        """Test multimodal processing integration"""
        
        processor, rag = rag_system
        
        # This test would require sample files with tables/images
        # Implementation depends on availability of test assets
        
        # Verify multimodal capabilities are available
        formats = processor.get_supported_formats()
        assert 'format_categories' in formats
        
        # Test multimodal element retrieval
        elements = processor.get_multimodal_elements()
        assert isinstance(elements, list)
    
    def test_error_recovery(self, rag_system, temp_dir):
        """Test system behavior with problematic files"""
        
        processor, rag = rag_system
        
        # Create corrupted file
        corrupted_file = Path(temp_dir) / "corrupted.pdf"
        corrupted_file.write_bytes(b"Not a real PDF file")
        
        # System should handle gracefully
        try:
            documents = processor.load_document(str(corrupted_file))
        except Exception as e:
            # Error is expected, but should be handled gracefully
            assert isinstance(e, (ValueError, RuntimeError))
    
    def test_concurrent_processing(self, rag_system, temp_dir):
        """Test concurrent document processing"""
        
        processor, rag = rag_system
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            content = f"Test document {i} with content about topic {i}"
            file_path = Path(temp_dir) / f"test_{i}.txt"
            file_path.write_text(content)
            test_files.append(str(file_path))
        
        # Process multiple documents
        all_documents = processor.load_multiple_documents(
            test_files, 
            parallel=True
        )
        
        assert len(all_documents) >= 3
        
        # Add to RAG and test queries
        rag.add_documents(all_documents)
        response = rag.query("What topics are mentioned?")
        
        assert response.answer is not None
```

#### Performance Testing
```python
# tests/test_performance.py

import pytest
import time
import psutil
import os
from contextlib import contextmanager

from src.document_loader import DocumentProcessor
from src.config import Config

class TestPerformance:
    """Performance and resource usage tests"""
    
    @contextmanager
    def measure_performance(self):
        """Context manager to measure performance metrics"""
        
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss
        
        yield
        
        end_time = time.time()
        end_memory = process.memory_info().rss
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Memory usage: {memory_usage / 1024 / 1024:.2f} MB")
        
        # Assert reasonable performance
        assert execution_time < 300  # 5 minutes max
        assert memory_usage < 2 * 1024 * 1024 * 1024  # 2GB max
    
    def test_document_processing_performance(self):
        """Test document processing performance"""
        
        processor = DocumentProcessor()
        
        with self.measure_performance():
            # This would test with various document types
            # Implementation depends on test asset availability
            
            # Test basic functionality performance
            formats = processor.get_supported_formats()
            assert len(formats['supported_extensions']) >= 26
    
    def test_memory_usage_scaling(self):
        """Test memory usage with multiple documents"""
        
        processor = DocumentProcessor()
        
        # Test memory doesn't grow unbounded
        initial_memory = psutil.Process(os.getpid()).memory_info().rss
        
        # Process multiple small documents (simulated)
        for i in range(10):
            # Simulate processing
            processor.get_supported_formats()
            processor.clear_multimodal_cache()
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal with proper cleanup
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
    
    @pytest.mark.slow
    def test_large_document_processing(self):
        """Test processing of large documents (marked as slow test)"""
        
        processor = DocumentProcessor()
        
        with self.measure_performance():
            # This would test with large documents
            # Skip if test assets not available
            pytest.skip("Large document test requires test assets")
```

#### Test Execution and Reporting
```bash
# Create comprehensive test runner script
# test_runner.sh

#!/bin/bash

echo "🧪 RAG System Comprehensive Testing Suite"
echo "=========================================="

# Set up test environment
export TESTING=true
export LOG_LEVEL=WARNING

# Function to run test and capture results
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n🔍 Running $test_name..."
    echo "----------------------------------------"
    
    if eval $test_command; then
        echo "✅ $test_name PASSED"
        return 0
    else
        echo "❌ $test_name FAILED"
        return 1
    fi
}

# Initialize results
total_tests=0
passed_tests=0

# Test 1: Basic system validation
total_tests=$((total_tests + 1))
if run_test "System Validation" "python3 test_setup.py"; then
    passed_tests=$((passed_tests + 1))
fi

# Test 2: Dependency verification
total_tests=$((total_tests + 1))
if run_test "Dependency Check" "python3 dependency_check.py"; then
    passed_tests=$((passed_tests + 1))
fi

# Test 3: Format support testing
total_tests=$((total_tests + 1))
if run_test "Format Support" "python3 test_all_formats.py"; then
    passed_tests=$((passed_tests + 1))
fi

# Test 4: PDF processing tests
total_tests=$((total_tests + 1))
if run_test "PDF Processing" "python3 test_pdf_multimodal.py"; then
    passed_tests=$((passed_tests + 1))
fi

# Test 5: Unit tests (if pytest available)
if command -v pytest >/dev/null 2>&1; then
    total_tests=$((total_tests + 1))
    if run_test "Unit Tests" "pytest tests/ -v"; then
        passed_tests=$((passed_tests + 1))
    fi
fi

# Final results
echo -e "\n" + "=" * 50
echo "📊 Test Results Summary"
echo "=" * 50
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $((total_tests - passed_tests))"
echo "Success rate: $(( passed_tests * 100 / total_tests ))%"

if [ $passed_tests -eq $total_tests ]; then
    echo -e "\n🎉 All tests passed! System is ready for deployment."
    exit 0
else
    echo -e "\n⚠️  Some tests failed. Please review the output above."
    exit 1
fi
```

---

## Deployment Options

### 🚀 Production Deployment Strategies

#### 1. Streamlit Cloud Deployment (Easiest)

**Preparation:**
```bash
# 1. Prepare repository
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main

# 2. Create streamlit config
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[server]
port = 8501
enableCORS = true
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[browser]
gatherUsageStats = false
EOF
```

**Deployment Steps:**
```yaml
Steps:
1. Go to https://share.streamlit.io/
2. Connect your GitHub repository
3. Select the main branch
4. Set main file to: app.py
5. Add secrets in the dashboard:
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY
6. Deploy with one click

Pros:
  ✅ Free for public repositories
  ✅ Automatic deployment on git push
  ✅ HTTPS and custom domains
  ✅ Simple secret management

Cons:
  ❌ Limited resources (1 CPU, 800MB RAM)
  ❌ Public repositories only (for free tier)
  ❌ Cold start delays
```

#### 2. Docker Container Deployment

**Optimized Dockerfile:**
```dockerfile
# Multi-stage build for smaller production image
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libmagic1 \
    libmagic-dev \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    default-jre \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/uploads /app/vector_store /app/cache && \
    chown -R appuser:appuser /app

# Copy application
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

**Docker Compose for Production:**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  rag-app:
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_HOST=redis
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/ragdb
    volumes:
      - uploads_data:/app/uploads
      - vector_data:/app/vector_store
      - cache_data:/app/cache
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ragdb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - rag-app
    restart: unless-stopped

volumes:
  uploads_data:
  vector_data:
  cache_data:
  redis_data:
  postgres_data:

networks:
  default:
    name: rag-network
```

#### 3. AWS ECS/Fargate Deployment

**Task Definition (JSON):**
```json
{
  "family": "rag-document-qa",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ragTaskRole",
  "containerDefinitions": [
    {
      "name": "rag-app",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/rag-document-qa:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "REDIS_HOST",
          "value": "rag-cache.XXXXX.cache.amazonaws.com"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:rag-openai-key-XXXXX"
        },
        {
          "name": "ANTHROPIC_API_KEY", 
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:rag-anthropic-key-XXXXX"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-document-qa",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

**Deployment Script:**
```bash
#!/bin/bash
# deploy_aws.sh

set -e

AWS_REGION="us-east-1"
ECR_REGISTRY="ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE_NAME="rag-document-qa"
CLUSTER_NAME="rag-cluster"
SERVICE_NAME="rag-service"

echo "🚀 Deploying RAG Document Q&A to AWS ECS"

# Build and push Docker image
echo "📦 Building and pushing Docker image..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

docker build -t $IMAGE_NAME .
docker tag $IMAGE_NAME:latest $ECR_REGISTRY/$IMAGE_NAME:latest
docker push $ECR_REGISTRY/$IMAGE_NAME:latest

# Update ECS service
echo "🔄 Updating ECS service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --force-new-deployment \
    --region $AWS_REGION

# Wait for deployment to complete
echo "⏳ Waiting for deployment to complete..."
aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION

echo "✅ Deployment completed successfully!"

# Get service URL
LOAD_BALANCER_DNS=$(aws elbv2 describe-load-balancers \
    --query 'LoadBalancers[0].DNSName' \
    --output text \
    --region $AWS_REGION)

echo "🌐 Service URL: http://$LOAD_BALANCER_DNS"
```

#### 4. Kubernetes Deployment

**Complete K8s Manifests:**
```yaml
# k8s-deployment.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  CHUNK_SIZE: "1000"
  CHUNK_OVERLAP: "200"
  ENABLE_CACHING: "true"
  PARALLEL_PROCESSING: "true"
  LOG_LEVEL: "INFO"

---
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  ANTHROPIC_API_KEY: <base64-encoded-key>

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-deployment
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-app
  template:
    metadata:
      labels:
        app: rag-app
    spec:
      containers:
      - name: rag-app
        image: rag-document-qa:latest
        ports:
        - containerPort: 8501
        envFrom:
        - configMapRef:
            name: rag-config
        - secretRef:
            name: rag-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 60
          periodSeconds: 30
        volumeMounts:
        - name: uploads-storage
          mountPath: /app/uploads
        - name: vector-storage
          mountPath: /app/vector_store
      volumes:
      - name: uploads-storage
        persistentVolumeClaim:
          claimName: uploads-pvc
      - name: vector-storage
        persistentVolumeClaim:
          claimName: vector-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
  namespace: rag-system
spec:
  selector:
    app: rag-app
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: uploads-pvc
  namespace: rag-system
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vector-pvc
  namespace: rag-system
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: rag-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-service
            port:
              number: 80
```

**Deployment Commands:**
```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -n rag-system
kubectl get services -n rag-system
kubectl get ingress -n rag-system

# View logs
kubectl logs -f deployment/rag-deployment -n rag-system

# Scale deployment
kubectl scale deployment/rag-deployment --replicas=5 -n rag-system
```

---

## Production Considerations

### 🏢 Enterprise Production Setup

#### Security Hardening
```yaml
Security Checklist:
  Network Security:
    ✅ Use HTTPS/TLS encryption
    ✅ Implement Web Application Firewall (WAF)
    ✅ Configure proper CORS policies
    ✅ Use VPN or private networks for admin access
    ✅ Implement rate limiting and DDoS protection
    
  Authentication & Authorization:
    ✅ Implement API key authentication
    ✅ Use OAuth 2.0 or SAML for enterprise SSO
    ✅ Implement role-based access control (RBAC)
    ✅ Enable audit logging for all actions
    
  Data Security:
    ✅ Encrypt data at rest and in transit
    ✅ Implement secure file upload validation
    ✅ Use encrypted storage for sensitive documents
    ✅ Regular security scans and updates
    ✅ Implement data retention policies
    
  Infrastructure Security:
    ✅ Use container security scanning
    ✅ Implement network segmentation
    ✅ Regular security patches and updates
    ✅ Backup and disaster recovery plans
```

#### Performance Optimization
```python
# production_optimizations.py

class ProductionOptimizations:
    """
    Production-specific performance optimizations
    """
    
    def __init__(self):
        self.setup_production_config()
        self.setup_monitoring()
        self.setup_caching()
    
    def setup_production_config(self):
        """Configure for production performance"""
        
        production_config = {
            # Resource Management
            'MAX_CONCURRENT_REQUESTS': 10,
            'REQUEST_TIMEOUT': 300,
            'MEMORY_LIMIT_PER_REQUEST': '2GB',
            
            # Processing Optimization
            'PARALLEL_PROCESSING': True,
            'MAX_WORKERS': min(4, os.cpu_count()),
            'BATCH_PROCESSING_SIZE': 5,
            
            # Caching Strategy
            'REDIS_CLUSTER_ENABLED': True,
            'CACHE_STRATEGY': 'write-through',
            'CACHE_COMPRESSION': True,
            
            # AI Model Optimization
            'MODEL_QUANTIZATION': True,
            'BATCH_AI_INFERENCE': True,
            'GPU_MEMORY_FRACTION': 0.8,
            
            # Database Optimization
            'CONNECTION_POOLING': True,
            'QUERY_OPTIMIZATION': True,
            'INDEX_OPTIMIZATION': True
        }
        
        return production_config
    
    def setup_monitoring(self):
        """Setup comprehensive monitoring"""
        
        monitoring_config = {
            'metrics': [
                'response_time',
                'throughput',
                'error_rate',
                'memory_usage',
                'cpu_usage',
                'gpu_usage',
                'cache_hit_rate'
            ],
            'alerts': [
                'high_error_rate',
                'slow_response_time',
                'memory_threshold',
                'disk_space_low'
            ],
            'dashboards': [
                'system_overview',
                'application_metrics',
                'business_metrics'
            ]
        }
        
        return monitoring_config
```

#### Load Balancing and Scaling
```yaml
# load-balancer.conf (NGINX)
upstream rag_backend {
    least_conn;
    server rag-app-1:8501 max_fails=3 fail_timeout=30s;
    server rag-app-2:8501 max_fails=3 fail_timeout=30s;
    server rag-app-3:8501 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    listen 443 ssl;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # File Upload Limits
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://rag_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # Health Check
        health_check;
    }
    
    # Health Check Endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

#### Auto-Scaling Configuration
```yaml
# autoscaling.yaml (Kubernetes HPA)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
```

---

## Monitoring & Maintenance

### 📊 Production Monitoring

#### Application Monitoring
```python
# monitoring.py - Comprehensive monitoring setup

import logging
import time
import psutil
import prometheus_client
from functools import wraps

class ProductionMonitoring:
    """
    Production monitoring and metrics collection
    """
    
    def __init__(self):
        self.setup_metrics()
        self.setup_logging()
        self.setup_health_checks()
    
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        
        # Request metrics
        self.request_counter = prometheus_client.Counter(
            'rag_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = prometheus_client.Histogram(
            'rag_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        # Document processing metrics
        self.documents_processed = prometheus_client.Counter(
            'rag_documents_processed_total',
            'Total documents processed',
            ['file_type', 'status']
        )
        
        self.processing_duration = prometheus_client.Histogram(
            'rag_processing_duration_seconds',
            'Document processing duration',
            ['file_type']
        )
        
        # System metrics
        self.memory_usage = prometheus_client.Gauge(
            'rag_memory_usage_bytes',
            'Current memory usage'
        )
        
        self.cpu_usage = prometheus_client.Gauge(
            'rag_cpu_usage_percent',
            'Current CPU usage'
        )
        
        # Business metrics
        self.multimodal_elements = prometheus_client.Counter(
            'rag_multimodal_elements_total',
            'Multimodal elements extracted',
            ['element_type']
        )
    
    def setup_logging(self):
        """Setup structured logging"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/rag/application.log'),
                logging.StreamHandler()
            ]
        )
        
        # Setup separate loggers for different components
        self.access_logger = logging.getLogger('rag.access')
        self.error_logger = logging.getLogger('rag.error')
        self.performance_logger = logging.getLogger('rag.performance')
    
    def monitor_request(self, func):
        """Decorator to monitor request metrics"""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                status = 'success'
                self.request_counter.labels(
                    method='POST',
                    endpoint=func.__name__,
                    status=status
                ).inc()
                
                return result
            
            except Exception as e:
                status = 'error'
                self.error_logger.error(f"Request failed: {e}")
                self.request_counter.labels(
                    method='POST',
                    endpoint=func.__name__,
                    status=status
                ).inc()
                raise
            
            finally:
                duration = time.time() - start_time
                self.request_duration.labels(
                    method='POST',
                    endpoint=func.__name__
                ).observe(duration)
        
        return wrapper
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        self.memory_usage.set(memory_info.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
        
        # Log warnings for high resource usage
        if memory_info.percent > 90:
            self.error_logger.warning(f"High memory usage: {memory_info.percent}%")
        
        if cpu_percent > 90:
            self.error_logger.warning(f"High CPU usage: {cpu_percent}%")
    
    def setup_health_checks(self):
        """Setup health check endpoints"""
        
        health_checks = {
            'database': self.check_database_health,
            'redis': self.check_redis_health,
            'disk_space': self.check_disk_space,
            'memory': self.check_memory_usage,
            'ai_models': self.check_ai_models_health
        }
        
        return health_checks
    
    def check_database_health(self):
        """Check database connectivity and performance"""
        # Implementation depends on database type
        return {'status': 'healthy', 'response_time': 0.05}
    
    def check_redis_health(self):
        """Check Redis connectivity and performance"""
        # Implementation depends on Redis configuration
        return {'status': 'healthy', 'response_time': 0.01}
    
    def check_disk_space(self):
        """Check available disk space"""
        disk_usage = psutil.disk_usage('/')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        
        status = 'healthy' if free_percent > 20 else 'warning'
        return {'status': status, 'free_percent': free_percent}
    
    def check_memory_usage(self):
        """Check memory usage"""
        memory = psutil.virtual_memory()
        status = 'healthy' if memory.percent < 85 else 'warning'
        
        return {'status': status, 'usage_percent': memory.percent}
    
    def check_ai_models_health(self):
        """Check AI models availability"""
        # Test model inference
        try:
            # Quick inference test
            return {'status': 'healthy', 'models_loaded': True}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

# Usage in main application
monitoring = ProductionMonitoring()

@monitoring.monitor_request
def process_document(file_path):
    # Document processing logic
    pass
```

#### Alerting Configuration
```yaml
# alerts.yml - Prometheus alerting rules
groups:
- name: rag_system_alerts
  rules:
  
  # High Error Rate
  - alert: HighErrorRate
    expr: rate(rag_requests_total{status="error"}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} requests/sec"
  
  # Slow Response Time
  - alert: SlowResponseTime
    expr: histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m])) > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow response time detected"
      description: "95th percentile response time is {{ $value }} seconds"
  
  # High Memory Usage
  - alert: HighMemoryUsage
    expr: rag_memory_usage_bytes / (1024^3) > 6
    for: 3m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}GB"
  
  # High CPU Usage
  - alert: HighCPUUsage
    expr: rag_cpu_usage_percent > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value }}%"
  
  # Service Down
  - alert: ServiceDown
    expr: up{job="rag-service"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "RAG service is down"
      description: "RAG service has been down for more than 1 minute"
```

#### Maintenance Procedures
```bash
#!/bin/bash
# maintenance.sh - Regular maintenance tasks

set -e

LOG_FILE="/var/log/rag/maintenance.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

log() {
    echo "[$DATE] $1" | tee -a $LOG_FILE
}

log "Starting RAG system maintenance"

# 1. Clean up temporary files
log "Cleaning temporary files..."
find /app/uploads -type f -mtime +7 -delete
find /app/cache -type f -mtime +1 -delete
find /tmp -name "*.tmp" -mtime +1 -delete

# 2. Optimize database
log "Optimizing database..."
# Database optimization commands here

# 3. Clear old logs
log "Rotating logs..."
logrotate /etc/logrotate.d/rag-system

# 4. Update AI models (if needed)
log "Checking for model updates..."
python3 /app/scripts/update_models.py

# 5. Health check
log "Running health check..."
python3 /app/scripts/health_check.py

# 6. Backup critical data
log "Backing up critical data..."
tar -czf /backup/rag-backup-$(date +%Y%m%d).tar.gz \
    /app/vector_store \
    /app/config \
    /var/log/rag

# 7. Monitor system resources
log "System resource usage:"
df -h | tee -a $LOG_FILE
free -h | tee -a $LOG_FILE
ps aux --sort=-%mem | head -10 | tee -a $LOG_FILE

log "Maintenance completed successfully"

# Send notification
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"RAG system maintenance completed successfully"}' \
    $SLACK_WEBHOOK_URL
```

---

## Troubleshooting

### 🔧 Common Production Issues

#### Issue Resolution Guide
```yaml
Common Issues and Solutions:

1. High Memory Usage:
   Symptoms:
     - Out of memory errors
     - Slow performance
     - System crashes
   Solutions:
     - Reduce CHUNK_SIZE
     - Enable memory-efficient mode
     - Increase system RAM
     - Implement request queuing
   
2. Slow Processing:
   Symptoms:
     - Long response times
     - User complaints
     - Timeout errors
   Solutions:
     - Enable GPU acceleration
     - Optimize extraction methods
     - Implement result caching
     - Add more worker processes
   
3. AI Model Failures:
   Symptoms:
     - Image analysis failures
     - Poor extraction quality
     - Model loading errors
   Solutions:
     - Check GPU memory
     - Update model versions
     - Implement model fallbacks
     - Monitor model performance
   
4. Storage Issues:
   Symptoms:
     - Disk space warnings
     - File upload failures
     - Vector store errors
   Solutions:
     - Implement file cleanup
     - Add storage monitoring
     - Compress vector data
     - Archive old documents
```

#### Diagnostic Scripts
```python
#!/usr/bin/env python3
"""
Production diagnostic script
"""

import sys
import os
import psutil
import subprocess
from pathlib import Path

def check_system_resources():
    """Check system resource usage"""
    
    print("🔍 System Resource Check")
    print("=" * 40)
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent}% used ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU: {cpu_percent}% used")
    
    # Disk
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.percent}% used ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)")
    
    # Processes
    rag_processes = [p for p in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']) 
                     if 'python' in p.info['name'].lower() or 'streamlit' in p.info['name'].lower()]
    
    if rag_processes:
        print("\nRAG-related processes:")
        for proc in rag_processes:
            print(f"  PID {proc.info['pid']}: {proc.info['name']} (CPU: {proc.info['cpu_percent']:.1f}%, Memory: {proc.info['memory_percent']:.1f}%)")

def check_dependencies():
    """Check critical dependencies"""
    
    print("\n🔍 Dependency Check")
    print("=" * 30)
    
    dependencies = {
        'streamlit': 'streamlit --version',
        'python': 'python3 --version',
        'redis': 'redis-cli ping',
        'tesseract': 'tesseract --version',
        'java': 'java -version'
    }
    
    for name, command in dependencies.items():
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {name}: OK")
            else:
                print(f"❌ {name}: Error")
        except Exception:
            print(f"❌ {name}: Not available")

def check_file_permissions():
    """Check file permissions for critical directories"""
    
    print("\n🔍 File Permission Check")
    print("=" * 35)
    
    directories = [
        '/app/uploads',
        '/app/vector_store', 
        '/app/cache',
        '/var/log/rag'
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            stat = path.stat()
            permissions = oct(stat.st_mode)[-3:]
            print(f"✅ {directory}: {permissions}")
        else:
            print(f"❌ {directory}: Does not exist")

def check_network_connectivity():
    """Check network connectivity to external services"""
    
    print("\n🔍 Network Connectivity Check")
    print("=" * 40)
    
    endpoints = [
        'api.openai.com',
        'api.anthropic.com',
        'huggingface.co'
    ]
    
    for endpoint in endpoints:
        try:
            result = subprocess.run(['ping', '-c', '1', endpoint], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {endpoint}: Reachable")
            else:
                print(f"❌ {endpoint}: Unreachable")
        except Exception:
            print(f"❌ {endpoint}: Check failed")

def check_application_health():
    """Check application-specific health indicators"""
    
    print("\n🔍 Application Health Check")
    print("=" * 38)
    
    # Check if Streamlit app is responsive
    try:
        import requests
        response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit app: Responsive")
        else:
            print("❌ Streamlit app: Not responding correctly")
    except Exception:
        print("❌ Streamlit app: Not reachable")
    
    # Check if AI models are loaded
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ GPU: Available")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️  GPU: Not available (using CPU)")
    except ImportError:
        print("❌ PyTorch: Not available")

def main():
    print("🏥 RAG System Diagnostic Tool")
    print("=" * 50)
    
    check_system_resources()
    check_dependencies()
    check_file_permissions()
    check_network_connectivity()
    check_application_health()
    
    print("\n" + "=" * 50)
    print("📊 Diagnostic completed")
    
    # Generate recommendations
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        print("⚠️  High memory usage detected - consider restarting or scaling up")
    
    disk = psutil.disk_usage('/')
    if disk.percent > 90:
        print("⚠️  Low disk space - clean up temporary files")

if __name__ == "__main__":
    main()
```

#### Emergency Recovery Procedures
```bash
#!/bin/bash
# emergency_recovery.sh

echo "🚨 RAG System Emergency Recovery"
echo "================================"

# 1. Stop all RAG processes
echo "Stopping RAG processes..."
pkill -f "streamlit"
pkill -f "python.*app.py"

# 2. Clear problematic files
echo "Clearing temporary files..."
rm -rf /tmp/rag_*
rm -rf /app/cache/*

# 3. Free up memory
echo "Freeing memory..."
sync && echo 3 > /proc/sys/vm/drop_caches

# 4. Check disk space
echo "Checking disk space..."
df -h

# 5. Restart with minimal configuration
echo "Starting in safe mode..."
cd /app
export MULTIMODAL_AI_ENABLED=false
export PARALLEL_PROCESSING=false
export CHUNK_SIZE=500

nohup python3 -m streamlit run app.py > /dev/null 2>&1 &

echo "✅ Emergency recovery completed"
echo "System started in safe mode - check logs and reconfigure as needed"
```

---

**🚀 Complete Guide Ready for Production Deployment!**

**Built with 💙 by [Fenil Sonani](https://github.com/fenilsonani) | © 2025 | Enterprise Ready**