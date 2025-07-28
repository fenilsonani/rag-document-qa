"""Setup script for RAG Document Q&A System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-document-qa",
    version="1.0.0",
    author="Fenil Sonani",
    description="RAG-based document Q&A system using LangChain and Streamlit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.20",
        "langchain-community>=0.0.38",
        "langchain-openai>=0.1.7",
        "langchain-anthropic>=0.1.11",
        "chromadb>=0.4.24",
        "pypdf>=4.2.0",
        "python-docx>=1.1.0",
        "unstructured>=0.13.7",
        "sentence-transformers>=2.7.0",
        "tiktoken>=0.7.0",
        "streamlit>=1.34.0",
        "streamlit-chat>=0.1.1",
        "python-dotenv>=1.0.1",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
    ],
    extras_require={
        "performance": ["faiss-cpu>=1.8.0"],
        "dev": ["pytest", "black", "flake8", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "rag-qa=app:main",
        ],
    },
)