#!/usr/bin/env python3
"""Test script to verify RAG system setup."""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import chromadb
        print(f"✅ ChromaDB: {chromadb.__version__}")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    return True

def test_src_modules():
    """Test if our source modules can be imported."""
    print("\n🏗️ Testing source modules...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        from src.config import Config
        print("✅ Config module")
        
        # Test config
        config = Config()
        print(f"   - Chunk size: {config.CHUNK_SIZE}")
        print(f"   - Supported extensions: {', '.join(config.SUPPORTED_EXTENSIONS)}")
        
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from src.document_loader import DocumentProcessor
        print("✅ Document Loader module")
    except ImportError as e:
        print(f"❌ Document Loader import failed: {e}")
        return False
    
    try:
        from src.vector_store import VectorStoreManager
        print("✅ Vector Store module")
    except ImportError as e:
        print(f"❌ Vector Store import failed: {e}")
        return False
    
    try:
        from src.embedding_service import EmbeddingService
        print("✅ Embedding Service module")
    except ImportError as e:
        print(f"❌ Embedding Service import failed: {e}")
        return False
    
    try:
        from src.rag_chain import RAGChain
        print("✅ RAG Chain module")
    except ImportError as e:
        print(f"❌ RAG Chain import failed: {e}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist."""
    print("\n📁 Testing directories...")
    
    required_dirs = ["src", "uploads", "vector_store", "docs"]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"⚠️  {dir_name}/ directory missing - creating...")
            dir_path.mkdir(exist_ok=True)
    
    return True

def test_env_setup():
    """Test environment setup."""
    print("\n🔧 Testing environment setup...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_example.exists():
        print("✅ .env.example file exists")
    else:
        print("❌ .env.example file missing")
        return False
    
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check for API keys (without revealing them)
        with open(env_file) as f:
            content = f.read()
            
        if "OPENAI_API_KEY=" in content:
            print("✅ OpenAI API key configuration found")
        if "ANTHROPIC_API_KEY=" in content:
            print("✅ Anthropic API key configuration found")
        
        if "OPENAI_API_KEY=" not in content and "ANTHROPIC_API_KEY=" not in content:
            print("⚠️  No API keys configured in .env file")
            
    else:
        print("⚠️  .env file missing - copy from .env.example and add your API keys")
    
    return True

def main():
    """Run all tests."""
    print("🧪 RAG System Setup Test")
    print("=" * 50)
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    tests = [
        test_directories,
        test_imports,
        test_src_modules,
        test_env_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG system is ready to use.")
        print("\n🚀 To start the application:")
        print("   ./run.sh")
        print("\n   Or manually:")
        print("   source venv/bin/activate")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())