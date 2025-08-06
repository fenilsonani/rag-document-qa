"""
Document Processing Service - Enterprise RAG Platform
Handles document ingestion, processing, and multi-modal analysis.
"""

import asyncio
import hashlib
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from uuid import uuid4

import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

from ..base.service_base import BaseService, ServiceStatus, ServiceRequest, ServiceResponse
from ...src.advanced_pdf_processor import AdvancedPDFProcessor
from ...src.universal_file_processor import UniversalFileProcessor
from ...src.multimodal_rag import MultiModalRAG


class DocumentProcessingRequest(ServiceRequest):
    """Document processing request model."""
    file_path: Optional[str] = None
    file_content: Optional[bytes] = None
    file_type: str
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    extract_tables: bool = True
    extract_images: bool = True
    enable_multimodal: bool = True
    chunk_strategy: str = "adaptive"  # adaptive, fixed, semantic
    quality_threshold: float = 0.8


class DocumentChunk(BaseModel):
    """Processed document chunk."""
    chunk_id: str
    content: str
    chunk_type: str  # text, table, image, chart
    page_number: Optional[int] = None
    position: Dict[str, float] = Field(default_factory=dict)  # x, y, width, height
    confidence_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embeddings: Optional[List[float]] = None


class ProcessedDocument(BaseModel):
    """Complete processed document result."""
    document_id: str
    filename: str
    file_type: str
    file_size: int
    total_pages: Optional[int] = None
    processing_time_ms: float
    chunks: List[DocumentChunk]
    extracted_tables: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_images: List[Dict[str, Any]] = Field(default_factory=list)
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    error_log: List[str] = Field(default_factory=list)


class DocumentProcessingService(BaseService):
    """
    Enterprise document processing service with advanced multi-modal capabilities.
    Handles PDF tables, images, charts, and 26+ file formats.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="document-processor",
            version="2.0.0",
            description="Enterprise Document Processing Service with Multi-Modal AI",
            port=8001,
            **kwargs
        )
        
        self.pdf_processor = None
        self.universal_processor = None
        self.multimodal_rag = None
        
        # Processing statistics
        self.processing_stats = {
            "documents_processed": 0,
            "total_chunks_generated": 0,
            "tables_extracted": 0,
            "images_processed": 0,
            "avg_processing_time_ms": 0.0,
            "error_count": 0
        }
        
        # Setup routes
        self._setup_document_routes()
    
    async def initialize(self):
        """Initialize document processing components."""
        try:
            self.logger.info("Initializing document processors...")
            
            # Initialize processors
            self.pdf_processor = AdvancedPDFProcessor()
            self.universal_processor = UniversalFileProcessor()
            self.multimodal_rag = MultiModalRAG()
            
            # Add circuit breakers for external services
            self.add_circuit_breaker("pdf_processing", failure_threshold=3, recovery_timeout=30.0)
            self.add_circuit_breaker("multimodal_ai", failure_threshold=5, recovery_timeout=60.0)
            
            self.logger.info("Document processors initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize document processing service: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup document processing resources."""
        self.logger.info("Shutting down document processing service...")
        # Cleanup any resources
        pass
    
    async def health_check(self) -> ServiceStatus:
        """Perform health check on document processing components."""
        try:
            # Test basic functionality
            if not self.pdf_processor or not self.universal_processor:
                return ServiceStatus.UNHEALTHY
            
            # Could add more sophisticated health checks
            return ServiceStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return ServiceStatus.UNHEALTHY
    
    def _setup_document_routes(self):
        """Setup document processing API routes."""
        
        @self.app.post("/api/v1/process/upload", response_model=ServiceResponse)
        async def process_uploaded_file(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            extract_tables: bool = True,
            extract_images: bool = True,
            enable_multimodal: bool = True,
            chunk_strategy: str = "adaptive",
            quality_threshold: float = 0.8
        ):
            """Process uploaded file and return structured document data."""
            request_id = str(uuid4())
            start_time = datetime.utcnow()
            
            try:
                async with self.trace_operation("process_uploaded_file"):
                    # Read file content
                    file_content = await file.read()
                    file_size = len(file_content)
                    
                    # Create processing request
                    processing_request = DocumentProcessingRequest(
                        request_id=request_id,
                        file_type=file.content_type or mimetypes.guess_type(file.filename)[0],
                        processing_options={
                            "extract_tables": extract_tables,
                            "extract_images": extract_images,
                            "enable_multimodal": enable_multimodal,
                            "chunk_strategy": chunk_strategy,
                            "quality_threshold": quality_threshold
                        },
                        file_content=file_content
                    )
                    
                    # Process document in background
                    background_tasks.add_task(
                        self._process_document_async,
                        processing_request,
                        file.filename,
                        file_size
                    )
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    return self.create_response(
                        request_id=request_id,
                        data={
                            "status": "processing_started",
                            "document_id": request_id,
                            "filename": file.filename,
                            "file_size": file_size
                        },
                        processing_time_ms=processing_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Error processing uploaded file: {e}")
                self.metrics.error_count += 1
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                return self.create_response(
                    request_id=request_id,
                    error=f"Processing failed: {str(e)}",
                    processing_time_ms=processing_time
                )
        
        @self.app.post("/api/v1/process/file", response_model=ServiceResponse)
        async def process_file_by_path(request: DocumentProcessingRequest):
            """Process file by file path."""
            start_time = datetime.utcnow()
            
            try:
                async with self.trace_operation("process_file_by_path"):
                    if not request.file_path or not os.path.exists(request.file_path):
                        raise HTTPException(status_code=400, detail="Invalid file path")
                    
                    file_path = Path(request.file_path)
                    file_size = file_path.stat().st_size
                    
                    # Read file content
                    async with aiofiles.open(file_path, 'rb') as f:
                        file_content = await f.read()
                    
                    # Process document
                    result = await self._process_document(
                        file_content=file_content,
                        filename=file_path.name,
                        file_type=request.file_type,
                        file_size=file_size,
                        options=request.processing_options,
                        request_id=request.request_id
                    )
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    return self.create_response(
                        request_id=request.request_id,
                        data=result.dict(),
                        processing_time_ms=processing_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Error processing file by path: {e}")
                self.metrics.error_count += 1
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                return self.create_response(
                    request_id=request.request_id,
                    error=f"Processing failed: {str(e)}",
                    processing_time_ms=processing_time
                )
        
        @self.app.get("/api/v1/document/{document_id}", response_model=ServiceResponse)
        async def get_document_status(document_id: str):
            """Get document processing status and results."""
            try:
                # Check cache for processing results
                cached_result = await self.cache_get(f"document:{document_id}")
                
                if cached_result:
                    return self.create_response(
                        request_id=document_id,
                        data=cached_result
                    )
                else:
                    return self.create_response(
                        request_id=document_id,
                        data={"status": "processing", "document_id": document_id}
                    )
                    
            except Exception as e:
                self.logger.error(f"Error getting document status: {e}")
                return self.create_response(
                    request_id=document_id,
                    error=f"Failed to get document status: {str(e)}"
                )
        
        @self.app.get("/api/v1/stats", response_model=ServiceResponse)
        async def get_processing_stats():
            """Get document processing statistics."""
            try:
                stats = self.processing_stats.copy()
                stats.update({
                    "service_uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                    "current_status": self.status,
                    "circuit_breaker_states": {
                        name: cb.state for name, cb in self.circuit_breakers.items()
                    }
                })
                
                return self.create_response(
                    request_id=str(uuid4()),
                    data=stats
                )
                
            except Exception as e:
                self.logger.error(f"Error getting stats: {e}")
                return self.create_response(
                    request_id=str(uuid4()),
                    error=f"Failed to get stats: {str(e)}"
                )
    
    async def _process_document_async(
        self,
        request: DocumentProcessingRequest,
        filename: str,
        file_size: int
    ):
        """Process document asynchronously and cache results."""
        try:
            result = await self._process_document(
                file_content=request.file_content,
                filename=filename,
                file_type=request.file_type,
                file_size=file_size,
                options=request.processing_options,
                request_id=request.request_id
            )
            
            # Cache the result
            await self.cache_set(f"document:{request.request_id}", result.dict(), ttl=3600)
            
            # Publish processing complete event
            await self.publish_event("document_processed", {
                "document_id": request.request_id,
                "filename": filename,
                "chunks_count": len(result.chunks),
                "processing_time_ms": result.processing_time_ms
            })
            
        except Exception as e:
            self.logger.error(f"Async document processing failed: {e}")
            error_result = {
                "status": "error",
                "error": str(e),
                "document_id": request.request_id
            }
            await self.cache_set(f"document:{request.request_id}", error_result, ttl=1800)
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
    async def _process_document(
        self,
        file_content: bytes,
        filename: str,
        file_type: str,
        file_size: int,
        options: Dict[str, Any],
        request_id: str
    ) -> ProcessedDocument:
        """Core document processing logic with circuit breaker protection."""
        start_time = datetime.utcnow()
        document_id = hashlib.sha256(file_content).hexdigest()
        
        try:
            chunks = []
            extracted_tables = []
            extracted_images = []
            error_log = []
            quality_metrics = {}
            
            # Determine processing strategy based on file type
            if file_type and "pdf" in file_type.lower():
                # Use advanced PDF processor
                try:
                    pdf_result = self.pdf_processor.process_pdf(
                        file_content,
                        extract_tables=options.get("extract_tables", True),
                        extract_images=options.get("extract_images", True),
                        enable_multimodal=options.get("enable_multimodal", True)
                    )
                    
                    # Convert PDF result to chunks
                    chunks.extend(self._convert_pdf_to_chunks(pdf_result, document_id))
                    extracted_tables = pdf_result.get("tables", [])
                    extracted_images = pdf_result.get("images", [])
                    quality_metrics = pdf_result.get("quality_metrics", {})
                    
                    self.processing_stats["tables_extracted"] += len(extracted_tables)
                    self.processing_stats["images_processed"] += len(extracted_images)
                    
                except Exception as e:
                    error_log.append(f"PDF processing error: {str(e)}")
                    self.logger.warning(f"PDF processing failed, falling back to universal processor: {e}")
            
            # Use universal file processor for other formats or as fallback
            if not chunks:
                try:
                    universal_result = self.universal_processor.process_file(
                        file_content,
                        filename,
                        file_type
                    )
                    
                    chunks.extend(self._convert_universal_to_chunks(universal_result, document_id))
                    
                except Exception as e:
                    error_log.append(f"Universal processing error: {str(e)}")
                    raise
            
            # Apply multi-modal enhancements if enabled
            if options.get("enable_multimodal", True) and extracted_images:
                try:
                    enhanced_chunks = await self._enhance_with_multimodal(chunks, extracted_images)
                    chunks.extend(enhanced_chunks)
                except Exception as e:
                    error_log.append(f"Multi-modal enhancement error: {str(e)}")
            
            # Filter chunks by quality threshold
            quality_threshold = options.get("quality_threshold", 0.8)
            chunks = [chunk for chunk in chunks if chunk.confidence_score >= quality_threshold]
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["total_chunks_generated"] += len(chunks)
            
            prev_avg = self.processing_stats["avg_processing_time_ms"]
            doc_count = self.processing_stats["documents_processed"]
            self.processing_stats["avg_processing_time_ms"] = (
                (prev_avg * (doc_count - 1) + processing_time) / doc_count
            )
            
            return ProcessedDocument(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                processing_time_ms=processing_time,
                chunks=chunks,
                extracted_tables=extracted_tables,
                extracted_images=extracted_images,
                quality_metrics=quality_metrics,
                error_log=error_log,
                document_metadata={
                    "processed_at": datetime.utcnow().isoformat(),
                    "processor_version": self.version,
                    "options": options
                }
            )
            
        except Exception as e:
            self.processing_stats["error_count"] += 1
            self.logger.error(f"Document processing failed: {e}")
            raise
    
    def _convert_pdf_to_chunks(self, pdf_result: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """Convert PDF processing results to standardized chunks."""
        chunks = []
        
        # Process text chunks
        for i, text_chunk in enumerate(pdf_result.get("text_chunks", [])):
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_text_{i}",
                content=text_chunk["content"],
                chunk_type="text",
                page_number=text_chunk.get("page", None),
                confidence_score=text_chunk.get("confidence", 0.9),
                metadata={
                    "extraction_method": "pdf_text",
                    "source": text_chunk.get("source", "unknown")
                }
            )
            chunks.append(chunk)
        
        # Process table chunks
        for i, table in enumerate(pdf_result.get("tables", [])):
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_table_{i}",
                content=table.get("content", str(table.get("data", ""))),
                chunk_type="table",
                page_number=table.get("page", None),
                position=table.get("bbox", {}),
                confidence_score=table.get("confidence", 0.85),
                metadata={
                    "extraction_method": table.get("method", "unknown"),
                    "table_structure": table.get("structure", {}),
                    "data": table.get("data", [])
                }
            )
            chunks.append(chunk)
        
        # Process image chunks
        for i, image in enumerate(pdf_result.get("images", [])):
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_image_{i}",
                content=image.get("description", ""),
                chunk_type="image",
                page_number=image.get("page", None),
                position=image.get("bbox", {}),
                confidence_score=image.get("confidence", 0.8),
                metadata={
                    "image_analysis": image.get("analysis", {}),
                    "ocr_text": image.get("ocr_text", ""),
                    "objects_detected": image.get("objects", [])
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _convert_universal_to_chunks(self, universal_result: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """Convert universal processor results to standardized chunks."""
        chunks = []
        
        for i, chunk_data in enumerate(universal_result.get("chunks", [])):
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_universal_{i}",
                content=chunk_data.get("content", ""),
                chunk_type=chunk_data.get("type", "text"),
                confidence_score=chunk_data.get("confidence", 0.9),
                metadata={
                    "extraction_method": "universal_processor",
                    "source_format": chunk_data.get("format", "unknown")
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _enhance_with_multimodal(
        self,
        chunks: List[DocumentChunk],
        images: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Enhance chunks with multi-modal AI analysis."""
        enhanced_chunks = []
        
        try:
            # Use circuit breaker for multimodal AI calls
            multimodal_cb = self.circuit_breakers.get("multimodal_ai")
            if multimodal_cb:
                multimodal_analysis = multimodal_cb(self.multimodal_rag.analyze_images)(images)
            else:
                multimodal_analysis = self.multimodal_rag.analyze_images(images)
            
            # Create enhanced chunks from multimodal analysis
            for i, analysis in enumerate(multimodal_analysis):
                enhanced_chunk = DocumentChunk(
                    chunk_id=f"multimodal_enhanced_{i}",
                    content=analysis.get("enhanced_description", ""),
                    chunk_type="multimodal",
                    confidence_score=analysis.get("confidence", 0.8),
                    metadata={
                        "multimodal_analysis": analysis,
                        "enhancement_type": "ai_vision"
                    }
                )
                enhanced_chunks.append(enhanced_chunk)
            
        except Exception as e:
            self.logger.warning(f"Multimodal enhancement failed: {e}")
        
        return enhanced_chunks


# Entry point for running the service
if __name__ == "__main__":
    service = DocumentProcessingService()
    service.run(debug=True)