"""Document loading and preprocessing functionality."""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from advanced_pdf_processor import AdvancedPDFProcessor
from multimodal_rag import MultiModalElement
from universal_file_processor import UniversalFileProcessor


class DocumentProcessor:
    """Handles document loading, processing, and chunking."""
    
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Initialize advanced PDF processor
        self.advanced_pdf_processor = AdvancedPDFProcessor(self.config)
        
        # Initialize universal file processor
        self.universal_processor = UniversalFileProcessor(self.config)
        
        # Store extracted multimodal elements
        self.multimodal_elements: List[MultiModalElement] = []
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on its file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")
        
        try:
            if extension == ".pdf":
                # Use advanced PDF processor for better table and image extraction
                documents, multimodal_elements = self.advanced_pdf_processor.process_pdf(file_path)
                
                # Store multimodal elements for later use
                self.multimodal_elements.extend(multimodal_elements)
                
                # If advanced processing fails, fallback to basic PyPDF
                if not documents:
                    print("Advanced PDF processing failed, falling back to basic PyPDFLoader")
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                
            elif extension in [".txt", ".docx", ".md"]:
                # Use traditional loaders for basic formats
                if extension == ".txt":
                    loader = TextLoader(file_path, encoding="utf-8")
                elif extension == ".docx":
                    loader = Docx2txtLoader(file_path)
                elif extension == ".md":
                    loader = UnstructuredMarkdownLoader(file_path)
                
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata.update({
                        "source": file_path,
                        "filename": path.name,
                        "file_type": extension
                    })
            
            else:
                # Use universal processor for all other formats
                print(f"🔄 Processing {extension} file with Universal Processor...")
                result = self.universal_processor.process_file(file_path)
                
                if result.success:
                    documents = result.documents
                    self.multimodal_elements.extend(result.multimodal_elements)
                    print(f"✅ Successfully processed {extension} file: {len(documents)} documents, {len(result.multimodal_elements)} multimodal elements")
                else:
                    print(f"❌ Universal processing failed: {result.error}")
                    # Fallback: try to read as text
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        documents = [Document(
                            page_content=content,
                            metadata={
                                "source": file_path,
                                "filename": path.name,
                                "file_type": extension,
                                "processing_method": "text_fallback"
                            }
                        )]
                        print(f"⚠️ Used text fallback for {extension} file")
                    except Exception as fallback_error:
                        print(f"❌ Text fallback also failed: {fallback_error}")
                        raise ValueError(f"Could not process {extension} file: {result.error}")
            
            return documents
            
        except Exception as e:
            raise RuntimeError(f"Error loading document {file_path}: {str(e)}")
    
    def load_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """Load multiple documents."""
        all_documents: List[Document] = []
        
        for file_path in file_paths:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {str(e)}")
                continue
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval."""
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Complete document processing pipeline."""
        # Load documents
        documents = self.load_multiple_documents(file_paths)
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        print(f"Processed {len(documents)} documents into {len(chunks)} chunks")
        
        return chunks
    
    def get_uploaded_files(self, upload_dir: Optional[str] = None) -> List[str]:
        """Get list of uploaded files."""
        upload_path = Path(upload_dir or self.config.UPLOAD_DIR)
        
        if not upload_path.exists():
            return []
        
        files = []
        for file_path in upload_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.config.SUPPORTED_EXTENSIONS:
                files.append(str(file_path))
        
        return files
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if a file can be processed."""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False
        
        # Check file extension
        if path.suffix.lower() not in self.config.SUPPORTED_EXTENSIONS:
            return False
        
        # Check file size (limit to 50MB)
        if path.stat().st_size > 50 * 1024 * 1024:
            return False
        
        return True
    
    def get_multimodal_elements(self) -> List[MultiModalElement]:
        """Get all extracted multimodal elements (tables, images, etc.)."""
        return self.multimodal_elements
    
    def get_tables(self) -> List[MultiModalElement]:
        """Get all extracted tables."""
        return [elem for elem in self.multimodal_elements if elem.element_type == "table"]
    
    def get_images(self) -> List[MultiModalElement]:
        """Get all extracted images."""
        return [elem for elem in self.multimodal_elements if elem.element_type in ["image", "chart"]]
    
    def get_processing_summary(self) -> dict:
        """Get a summary of processed multimodal elements."""
        summary = {
            "total_multimodal_elements": len(self.multimodal_elements),
            "tables": len(self.get_tables()),
            "images": len(self.get_images()),
            "extraction_methods": list(set(elem.processing_method for elem in self.multimodal_elements)),
            "avg_confidence": sum(elem.confidence_score for elem in self.multimodal_elements) / len(self.multimodal_elements) if self.multimodal_elements else 0
        }
        
        # Add PDF processor summary if available
        if hasattr(self, 'advanced_pdf_processor'):
            summary["pdf_processing"] = self.advanced_pdf_processor.get_processing_summary()
        
        return summary
    
    def clear_multimodal_cache(self):
        """Clear cached multimodal elements to free memory."""
        self.multimodal_elements.clear()
        if hasattr(self, 'advanced_pdf_processor'):
            self.advanced_pdf_processor.clear_cache()
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get comprehensive information about all supported file formats."""
        base_info = {
            "total_supported_extensions": len(self.config.SUPPORTED_EXTENSIONS),
            "supported_extensions": self.config.SUPPORTED_EXTENSIONS
        }
        
        # Get detailed format information from universal processor
        if hasattr(self, 'universal_processor'):
            format_details = self.universal_processor.get_supported_formats()
            base_info["format_categories"] = format_details
        
        return base_info
    
    def validate_file_support(self, file_path: str) -> Dict[str, Any]:
        """Check if a file is supported and what features are available."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        validation_result = {
            "file_path": file_path,
            "filename": path.name,
            "extension": extension,
            "is_supported": extension in self.config.SUPPORTED_EXTENSIONS,
            "file_exists": path.exists(),
            "processing_method": None,
            "available_features": [],
            "estimated_processing_time": "unknown"
        }
        
        if not validation_result["is_supported"]:
            validation_result["error"] = f"Unsupported file type: {extension}"
            return validation_result
        
        if not validation_result["file_exists"]:
            validation_result["error"] = "File not found"
            return validation_result
        
        # Determine processing method and features
        if extension == ".pdf":
            validation_result["processing_method"] = "advanced_pdf_processor"
            validation_result["available_features"] = [
                "text_extraction", "table_extraction", "image_extraction",
                "layout_analysis", "multi_page_support"
            ]
            validation_result["estimated_processing_time"] = "30-120 seconds"
        
        elif extension in [".txt", ".docx", ".md"]:
            validation_result["processing_method"] = "langchain_loaders"
            validation_result["available_features"] = ["text_extraction", "basic_metadata"]
            validation_result["estimated_processing_time"] = "5-30 seconds"
        
        elif extension in [".xlsx", ".xls", ".csv"]:
            validation_result["processing_method"] = "universal_processor_excel"
            validation_result["available_features"] = [
                "multi_sheet_extraction", "table_conversion", "data_analysis"
            ]
            validation_result["estimated_processing_time"] = "10-60 seconds"
        
        elif extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
            validation_result["processing_method"] = "universal_processor_image"
            validation_result["available_features"] = [
                "ocr_text_extraction", "ai_image_analysis", "object_detection"
            ]
            validation_result["estimated_processing_time"] = "15-45 seconds"
        
        elif extension in [".pptx", ".ppt"]:
            validation_result["processing_method"] = "universal_processor_powerpoint"
            validation_result["available_features"] = [
                "slide_text_extraction", "table_extraction", "image_detection"
            ]
            validation_result["estimated_processing_time"] = "20-90 seconds"
        
        else:
            validation_result["processing_method"] = "universal_processor_generic"
            validation_result["available_features"] = ["text_extraction", "structure_parsing"]
            validation_result["estimated_processing_time"] = "5-30 seconds"
        
        # Check file size
        try:
            file_size = path.stat().st_size
            validation_result["file_size_mb"] = round(file_size / (1024 * 1024), 2)
            
            if file_size > 50 * 1024 * 1024:  # 50MB
                validation_result["warning"] = "Large file size may increase processing time"
        except:
            pass
        
        return validation_result