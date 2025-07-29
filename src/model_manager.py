"""
Model Manager - Handles offline model downloading with progress tracking
Provides model selection and download management for the RAG system.
"""

import os
import json
import logging
import time
import streamlit as st
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import hashlib

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Model downloading will be limited.")

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available. Direct model downloading will be limited.")


@dataclass
class ModelInfo:
    """Information about an available model."""
    model_id: str
    name: str
    description: str
    task: str
    size_mb: int
    download_url: str
    local_path: Optional[str] = None
    is_downloaded: bool = False
    performance_score: float = 0.0
    memory_usage: str = "Unknown"
    supported_languages: List[str] = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["English"]


class ModelDownloadProgress:
    """Tracks download progress for a model."""
    
    def __init__(self, model_id: str, total_size: int = 0):
        self.model_id = model_id
        self.total_size = total_size
        self.downloaded_size = 0
        self.start_time = datetime.now()
        self.status = "initializing"  # initializing, downloading, extracting, completed, error
        self.error_message = ""
        self.files_downloaded = 0
        self.total_files = 0
        
    def update_progress(self, downloaded: int, total: int = None):
        """Update download progress."""
        self.downloaded_size = downloaded
        if total:
            self.total_size = total
        
    def get_progress_percentage(self) -> float:
        """Get download progress as percentage."""
        if self.total_size == 0:
            return 0.0
        return min(100.0, (self.downloaded_size / self.total_size) * 100)
    
    def get_speed_mbps(self) -> float:
        """Get download speed in MB/s."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return 0.0
        return (self.downloaded_size / (1024 * 1024)) / elapsed
    
    def get_eta_seconds(self) -> float:
        """Get estimated time to completion in seconds."""
        speed = self.get_speed_mbps()
        if speed == 0:
            return 0.0
        remaining_mb = (self.total_size - self.downloaded_size) / (1024 * 1024)
        return remaining_mb / speed


class ModelManager:
    """Manages offline model downloading and selection."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.available_models = self._initialize_available_models()
        self.download_progress = {}
        
    def _initialize_available_models(self) -> Dict[str, ModelInfo]:
        """Initialize the catalog of available models."""
        models = {
            "qa_distilbert": ModelInfo(
                model_id="distilbert-base-cased-distilled-squad",
                name="DistilBERT QA (Recommended)",
                description="Fast and efficient question-answering model, good balance of speed and accuracy",
                task="question-answering",
                size_mb=250,
                download_url="https://huggingface.co/distilbert-base-cased-distilled-squad",
                performance_score=8.5,
                memory_usage="~500MB",
                supported_languages=["English"]
            ),
            "qa_roberta": ModelInfo(
                model_id="deepset/roberta-base-squad2",
                name="RoBERTa QA (High Accuracy)",
                description="Higher accuracy question-answering model, slower but more precise",
                task="question-answering",
                size_mb=440,
                download_url="https://huggingface.co/deepset/roberta-base-squad2",
                performance_score=9.2,
                memory_usage="~800MB",
                supported_languages=["English"]
            ),
            "qa_electra": ModelInfo(
                model_id="google/electra-small-discriminator",
                name="ELECTRA QA (Lightweight)",
                description="Lightweight question-answering model, very fast with good accuracy",
                task="question-answering",
                size_mb=50,
                download_url="https://huggingface.co/google/electra-small-discriminator",
                performance_score=7.8,
                memory_usage="~200MB",
                supported_languages=["English"]
            ),
            "summarizer_bart": ModelInfo(
                model_id="facebook/bart-large-cnn",
                name="BART Summarizer (Recommended)",
                description="High-quality text summarization model",
                task="summarization",
                size_mb=1600,
                download_url="https://huggingface.co/facebook/bart-large-cnn",
                performance_score=9.0,
                memory_usage="~3GB",
                supported_languages=["English"]
            ),
            "summarizer_t5": ModelInfo(
                model_id="t5-small",
                name="T5-Small Summarizer (Lightweight)",
                description="Lightweight summarization model, good for resource-constrained environments",
                task="summarization",
                size_mb=240,
                download_url="https://huggingface.co/t5-small",
                performance_score=7.5,
                memory_usage="~500MB",
                supported_languages=["English"]
            ),
            "embedder_sentence": ModelInfo(
                model_id="all-MiniLM-L6-v2",
                name="Sentence Transformer (Recommended)",
                description="Fast and accurate sentence embeddings for semantic search",
                task="embedding",
                size_mb=90,
                download_url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                performance_score=8.8,
                memory_usage="~300MB",
                supported_languages=["100+ languages"]
            ),
            "embedder_multilingual": ModelInfo(
                model_id="paraphrase-multilingual-MiniLM-L12-v2",
                name="Multilingual Embedder",
                description="Multilingual sentence embeddings supporting 50+ languages",
                task="embedding",
                size_mb=420,
                download_url="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                performance_score=8.5,
                memory_usage="~600MB",
                supported_languages=["50+ languages"]
            )
        }
        
        # Check which models are already downloaded
        for model_key, model_info in models.items():
            model_path = self.models_dir / model_info.model_id.replace("/", "_")
            if model_path.exists() and any(model_path.iterdir()):
                model_info.is_downloaded = True
                model_info.local_path = str(model_path)
        
        return models
    
    def get_models_by_task(self, task: str) -> Dict[str, ModelInfo]:
        """Get models filtered by task type."""
        return {k: v for k, v in self.available_models.items() if v.task == task}
    
    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.available_models.get(model_key)
    
    def is_model_downloaded(self, model_key: str) -> bool:
        """Check if a model is already downloaded."""
        model_info = self.get_model_info(model_key)
        if not model_info:
            return False
        return model_info.is_downloaded
    
    def get_model_path(self, model_key: str) -> Optional[str]:
        """Get the local path of a downloaded model."""
        model_info = self.get_model_info(model_key)
        if not model_info or not model_info.is_downloaded:
            return None
        return model_info.local_path
    
    def estimate_total_download_size(self, model_keys: List[str]) -> int:
        """Estimate total download size for selected models."""
        total_size = 0
        for model_key in model_keys:
            model_info = self.get_model_info(model_key)
            if model_info and not model_info.is_downloaded:
                total_size += model_info.size_mb
        return total_size
    
    def download_model(
        self, 
        model_key: str, 
        progress_callback: Optional[Callable[[ModelDownloadProgress], None]] = None
    ) -> Dict[str, Any]:
        """Download a model with progress tracking."""
        model_info = self.get_model_info(model_key)
        if not model_info:
            return {"success": False, "error": f"Model {model_key} not found"}
        
        if model_info.is_downloaded:
            return {"success": True, "message": "Model already downloaded", "path": model_info.local_path}
        
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "transformers library not available"}
        
        # Initialize progress tracking
        progress = ModelDownloadProgress(model_info.model_id, model_info.size_mb * 1024 * 1024)
        self.download_progress[model_key] = progress
        
        try:
            # Create model directory
            model_path = self.models_dir / model_info.model_id.replace("/", "_")
            model_path.mkdir(exist_ok=True)
            
            progress.status = "downloading"
            
            # Update progress during download
            if progress_callback:
                progress_callback(progress)
            
            # Download using transformers
            if model_info.task == "question-answering":
                # Update progress - downloading tokenizer
                progress.downloaded_size = int(progress.total_size * 0.2)
                if progress_callback:
                    progress_callback(progress)
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info.model_id,
                    cache_dir=str(model_path)
                )
                
                # Update progress - downloading model
                progress.downloaded_size = int(progress.total_size * 0.7)
                if progress_callback:
                    progress_callback(progress)
                
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_info.model_id,
                    cache_dir=str(model_path)
                )
                
                # Update progress - saving locally
                progress.downloaded_size = int(progress.total_size * 0.9)
                if progress_callback:
                    progress_callback(progress)
                
                # Save locally
                tokenizer.save_pretrained(str(model_path))
                model.save_pretrained(str(model_path))
                
            elif model_info.task == "summarization":
                # Update progress - downloading pipeline
                progress.downloaded_size = int(progress.total_size * 0.5)
                if progress_callback:
                    progress_callback(progress)
                
                # Use pipeline for summarization models
                summarizer = pipeline(
                    "summarization",
                    model=model_info.model_id,
                    tokenizer=model_info.model_id,
                    cache_dir=str(model_path)
                )
                
                # Update progress - saving pipeline
                progress.downloaded_size = int(progress.total_size * 0.9)
                if progress_callback:
                    progress_callback(progress)
                
                # Save pipeline components
                summarizer.save_pretrained(str(model_path))
                
            elif model_info.task == "embedding":
                # Update progress - downloading embedder
                progress.downloaded_size = int(progress.total_size * 0.3)
                if progress_callback:
                    progress_callback(progress)
                
                # For sentence transformers
                try:
                    from sentence_transformers import SentenceTransformer
                    
                    # Update progress - loading model
                    progress.downloaded_size = int(progress.total_size * 0.7)
                    if progress_callback:
                        progress_callback(progress)
                    
                    model = SentenceTransformer(model_info.model_id, cache_folder=str(self.models_dir))
                    
                    # Update progress - saving model
                    progress.downloaded_size = int(progress.total_size * 0.9)
                    if progress_callback:
                        progress_callback(progress)
                    
                    model.save(str(model_path))
                except ImportError:
                    # Fallback to regular transformers
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_info.model_id,
                        cache_dir=str(model_path)
                    )
                    
                    # Update progress - downloading model
                    progress.downloaded_size = int(progress.total_size * 0.7)
                    if progress_callback:
                        progress_callback(progress)
                    
                    model = AutoModel.from_pretrained(
                        model_info.model_id,
                        cache_dir=str(model_path)
                    )
                    
                    # Update progress - saving components
                    progress.downloaded_size = int(progress.total_size * 0.9)
                    if progress_callback:
                        progress_callback(progress)
                    
                    tokenizer.save_pretrained(str(model_path))
                    model.save_pretrained(str(model_path))
            
            progress.status = "completed"
            progress.downloaded_size = progress.total_size
            
            # Update model info
            model_info.is_downloaded = True
            model_info.local_path = str(model_path)
            
            # Save model registry
            self._save_model_registry()
            
            if progress_callback:
                progress_callback(progress)
            
            return {
                "success": True,
                "message": f"Model {model_info.name} downloaded successfully",
                "path": str(model_path),
                "size_mb": model_info.size_mb
            }
            
        except Exception as e:
            progress.status = "error"
            progress.error_message = str(e)
            
            if progress_callback:
                progress_callback(progress)
            
            return {
                "success": False,
                "error": f"Failed to download {model_info.name}: {str(e)}"
            }
    
    def download_multiple_models(
        self,
        model_keys: List[str],
        progress_callback: Optional[Callable[[str, ModelDownloadProgress], None]] = None
    ) -> Dict[str, Any]:
        """Download multiple models with combined progress tracking."""
        results = {}
        failed_models = []
        
        for model_key in model_keys:
            def individual_callback(progress: ModelDownloadProgress):
                if progress_callback:
                    progress_callback(model_key, progress)
            
            result = self.download_model(model_key, individual_callback)
            results[model_key] = result
            
            if not result["success"]:
                failed_models.append(model_key)
        
        return {
            "success": len(failed_models) == 0,
            "results": results,
            "failed_models": failed_models,
            "total_downloaded": len(model_keys) - len(failed_models)
        }
    
    def delete_model(self, model_key: str) -> Dict[str, Any]:
        """Delete a downloaded model."""
        model_info = self.get_model_info(model_key)
        if not model_info:
            return {"success": False, "error": f"Model {model_key} not found"}
        
        if not model_info.is_downloaded:
            return {"success": True, "message": "Model not downloaded"}
        
        try:
            model_path = Path(model_info.local_path)
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            
            # Update model info
            model_info.is_downloaded = False
            model_info.local_path = None
            
            # Save updated registry
            self._save_model_registry()
            
            return {"success": True, "message": f"Model {model_info.name} deleted successfully"}
            
        except Exception as e:
            return {"success": False, "error": f"Failed to delete model: {str(e)}"}
    
    def get_download_progress(self, model_key: str) -> Optional[ModelDownloadProgress]:
        """Get current download progress for a model."""
        return self.download_progress.get(model_key)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information for downloaded models."""
        total_size = 0
        downloaded_models = 0
        
        for model_info in self.available_models.values():
            if model_info.is_downloaded:
                total_size += model_info.size_mb
                downloaded_models += 1
        
        return {
            "total_models_available": len(self.available_models),
            "downloaded_models": downloaded_models,
            "total_size_mb": total_size,
            "models_dir": str(self.models_dir),
            "storage_usage": self._get_directory_size_mb(self.models_dir)
        }
    
    def _get_directory_size_mb(self, path: Path) -> float:
        """Get directory size in MB."""
        try:
            total_size = 0
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _save_model_registry(self):
        """Save model registry to disk."""
        try:
            registry_path = self.models_dir / "model_registry.json"
            registry_data = {}
            
            for model_key, model_info in self.available_models.items():
                registry_data[model_key] = {
                    "model_id": model_info.model_id,
                    "is_downloaded": model_info.is_downloaded,
                    "local_path": model_info.local_path,
                    "download_date": datetime.now().isoformat() if model_info.is_downloaded else None
                }
            
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Failed to save model registry: {e}")
    
    def load_model_registry(self):
        """Load model registry from disk."""
        try:
            registry_path = self.models_dir / "model_registry.json"
            if not registry_path.exists():
                return
            
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            for model_key, data in registry_data.items():
                if model_key in self.available_models:
                    model_info = self.available_models[model_key]
                    model_info.is_downloaded = data.get("is_downloaded", False)
                    model_info.local_path = data.get("local_path")
                    
        except Exception as e:
            logging.warning(f"Failed to load model registry: {e}")
    
    def get_recommended_models(self) -> List[str]:
        """Get recommended model combination for new users."""
        return [
            "qa_distilbert",      # Fast QA model
            "summarizer_t5",      # Lightweight summarizer
            "embedder_sentence"   # Good embeddings
        ]
    
    def get_high_performance_models(self) -> List[str]:
        """Get high-performance model combination."""
        return [
            "qa_roberta",         # High accuracy QA
            "summarizer_bart",    # Best summarizer
            "embedder_sentence"   # Good embeddings
        ]
    
    def get_lightweight_models(self) -> List[str]:
        """Get lightweight model combination for limited resources."""
        return [
            "qa_electra",         # Lightweight QA
            "summarizer_t5",      # Small summarizer
            "embedder_sentence"   # Efficient embeddings
        ]