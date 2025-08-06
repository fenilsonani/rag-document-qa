"""
Multi-Modal RAG System - Pro-Level Enhancement
Implements intelligent processing of tables, images, charts, and other visual elements in documents.
"""

import re
import base64
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import tempfile
import io

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas not available. Table processing will be limited.")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image processing will be limited.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Advanced image processing will be limited.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. OCR functionality will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available. Chart analysis will be limited.")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Advanced image understanding will be limited.")

from langchain.schema import Document
from .config import Config


@dataclass
class MultiModalElement:
    """Represents a multi-modal element (table, image, chart, etc.)."""
    element_id: str
    element_type: str  # table, image, chart, diagram, formula, etc.
    content: Any  # DataFrame for tables, PIL Image for images, etc.
    metadata: Dict[str, Any]
    text_description: str
    structured_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    processing_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Handle non-serializable content
        if self.element_type == "table" and PANDAS_AVAILABLE and isinstance(self.content, pd.DataFrame):
            result["content"] = self.content.to_dict()
        elif self.element_type in ["image", "chart", "diagram"] and PIL_AVAILABLE and isinstance(self.content, Image.Image):
            # Convert image to base64 for serialization
            buffer = io.BytesIO()
            self.content.save(buffer, format='PNG')
            buffer.seek(0)
            result["content"] = base64.b64encode(buffer.read()).decode()
        elif hasattr(self.content, 'tolist'):  # numpy arrays
            result["content"] = self.content.tolist()
        else:
            result["content"] = str(self.content) if self.content is not None else None
        
        return result


@dataclass
class TableAnalysis:
    """Analysis results for a table."""
    num_rows: int
    num_columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    summary_stats: Dict[str, Any]
    key_insights: List[str]
    relationships: List[Dict[str, str]]
    
    
@dataclass
class ImageAnalysis:
    """Analysis results for an image."""
    width: int
    height: int
    format: str
    color_mode: str
    dominant_colors: List[Tuple[int, int, int]]
    detected_text: str
    description: str
    objects_detected: List[Dict[str, Any]]
    chart_type: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None


class TableProcessor:
    """Advanced table processing and analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
    
    def extract_tables_from_html(self, html_content: str) -> List[MultiModalElement]:
        """Extract tables from HTML content."""
        if not PANDAS_AVAILABLE:
            return []
        
        try:
            # Use pandas to read HTML tables
            tables = pd.read_html(html_content)
            elements = []
            
            for i, table in enumerate(tables):
                # Analyze table
                analysis = self._analyze_table(table)
                
                # Create text description
                description = self._generate_table_description(table, analysis)
                
                element = MultiModalElement(
                    element_id=f"table_{i}",
                    element_type="table",
                    content=table,
                    metadata={
                        "extraction_method": "html_pandas",
                        "table_index": i,
                        "analysis": asdict(analysis)
                    },
                    text_description=description,
                    structured_data=table.to_dict(),
                    confidence_score=0.9,
                    processing_method="pandas_html"
                )
                
                elements.append(element)
            
            return elements
            
        except Exception as e:
            logging.warning(f"Table extraction from HTML failed: {e}")
            return []
    
    def extract_tables_from_text(self, text: str) -> List[MultiModalElement]:
        """Extract tables from plain text using pattern recognition."""
        elements = []
        
        # Look for table-like patterns
        lines = text.split('\n')
        table_candidates = []
        current_table = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_table:
                    table_candidates.append(current_table)
                    current_table = []
                continue
            
            # Check if line looks like a table row
            if self._is_table_row(line):
                current_table.append(line)
            else:
                if current_table:
                    table_candidates.append(current_table)
                    current_table = []
        
        # Process table candidates
        for i, table_lines in enumerate(table_candidates):
            if len(table_lines) >= 2:  # At least header + one row
                try:
                    df = self._parse_text_table(table_lines)
                    if df is not None and not df.empty:
                        analysis = self._analyze_table(df)
                        description = self._generate_table_description(df, analysis)
                        
                        element = MultiModalElement(
                            element_id=f"text_table_{i}",
                            element_type="table",
                            content=df,
                            metadata={
                                "extraction_method": "text_pattern",
                                "original_lines": table_lines,
                                "analysis": asdict(analysis)
                            },
                            text_description=description,
                            structured_data=df.to_dict(),
                            confidence_score=0.7,
                            processing_method="text_parsing"
                        )
                        
                        elements.append(element)
                
                except Exception as e:
                    logging.warning(f"Failed to parse text table {i}: {e}")
        
        return elements
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row."""
        # Look for common table separators
        separators = ['|', '\t', '  ', ',']
        separator_counts = [line.count(sep) for sep in separators]
        
        # If any separator appears multiple times, likely a table row
        return any(count >= 2 for count in separator_counts)
    
    def _parse_text_table(self, lines: List[str]) -> Optional[pd.DataFrame]:
        """Parse text table into DataFrame."""
        if not PANDAS_AVAILABLE:
            return None
        
        try:
            # Try different separators
            separators = ['|', '\t', ',']
            
            for sep in separators:
                # Check if this separator works consistently
                row_lengths = []
                for line in lines:
                    if sep == '|':
                        # Special handling for markdown-style tables
                        parts = [part.strip() for part in line.split(sep) if part.strip()]
                    else:
                        parts = [part.strip() for part in line.split(sep)]
                    
                    if len(parts) > 1:
                        row_lengths.append(len(parts))
                
                # If most rows have the same number of columns, use this separator
                if row_lengths and len(set(row_lengths)) <= 2:  # Allow some variation
                    most_common_length = max(set(row_lengths), key=row_lengths.count)
                    
                    # Parse table
                    data = []
                    for line in lines:
                        if sep == '|':
                            parts = [part.strip() for part in line.split(sep) if part.strip()]
                        else:
                            parts = [part.strip() for part in line.split(sep)]
                        
                        if len(parts) == most_common_length:
                            data.append(parts)
                    
                    if len(data) >= 2:
                        # First row as header
                        df = pd.DataFrame(data[1:], columns=data[0])
                        return df
            
            return None
            
        except Exception as e:
            logging.warning(f"Text table parsing failed: {e}")
            return None
    
    def _analyze_table(self, df: pd.DataFrame) -> TableAnalysis:
        """Analyze table structure and content."""
        try:
            # Basic statistics
            num_rows, num_columns = df.shape
            column_names = df.columns.tolist()
            
            # Data types
            data_types = {}
            for col in df.columns:
                # Try to infer better data types
                dtype_str = str(df[col].dtype)
                if dtype_str == 'object':
                    # Check if numeric
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        data_types[col] = 'numeric'
                    except:
                        data_types[col] = 'text'
                else:
                    data_types[col] = dtype_str
            
            # Summary statistics for numeric columns
            summary_stats = {}
            numeric_cols = [col for col, dtype in data_types.items() if 'numeric' in dtype or 'int' in dtype or 'float' in dtype]
            
            for col in numeric_cols:
                try:
                    col_data = pd.to_numeric(df[col], errors='coerce')
                    summary_stats[col] = {
                        'mean': float(col_data.mean()) if not col_data.isna().all() else None,
                        'median': float(col_data.median()) if not col_data.isna().all() else None,
                        'min': float(col_data.min()) if not col_data.isna().all() else None,
                        'max': float(col_data.max()) if not col_data.isna().all() else None,
                        'std': float(col_data.std()) if not col_data.isna().all() else None
                    }
                except:
                    pass
            
            # Generate key insights
            insights = []
            insights.append(f"Table contains {num_rows} rows and {num_columns} columns")
            
            if numeric_cols:
                insights.append(f"Numeric columns: {', '.join(numeric_cols)}")
            
            text_cols = [col for col, dtype in data_types.items() if dtype == 'text']
            if text_cols:
                insights.append(f"Text columns: {', '.join(text_cols)}")
            
            # Look for relationships
            relationships = []
            if len(numeric_cols) >= 2:
                relationships.append({
                    "type": "correlation_candidates",
                    "columns": numeric_cols,
                    "description": "Numeric columns that could be analyzed for correlations"
                })
            
            return TableAnalysis(
                num_rows=num_rows,
                num_columns=num_columns,
                column_names=column_names,
                data_types=data_types,
                summary_stats=summary_stats,
                key_insights=insights,
                relationships=relationships
            )
            
        except Exception as e:
            logging.warning(f"Table analysis failed: {e}")
            return TableAnalysis(
                num_rows=0,
                num_columns=0,
                column_names=[],
                data_types={},
                summary_stats={},
                key_insights=["Analysis failed"],
                relationships=[]
            )
    
    def _generate_table_description(self, df: pd.DataFrame, analysis: TableAnalysis) -> str:
        """Generate human-readable description of table."""
        try:
            description_parts = []
            
            # Basic structure
            description_parts.append(f"This table has {analysis.num_rows} rows and {analysis.num_columns} columns.")
            
            # Column information
            if analysis.column_names:
                description_parts.append(f"The columns are: {', '.join(analysis.column_names)}.")
            
            # Data types
            numeric_cols = [col for col, dtype in analysis.data_types.items() if 'numeric' in dtype]
            text_cols = [col for col, dtype in analysis.data_types.items() if dtype == 'text']
            
            if numeric_cols:
                description_parts.append(f"Numeric data columns: {', '.join(numeric_cols)}.")
            
            if text_cols:
                description_parts.append(f"Text data columns: {', '.join(text_cols)}.")
            
            # Key statistics
            for col, stats in analysis.summary_stats.items():
                if stats and stats.get('mean') is not None:
                    description_parts.append(
                        f"Column '{col}' has values ranging from {stats['min']:.2f} to {stats['max']:.2f} "
                        f"with an average of {stats['mean']:.2f}."
                    )
            
            # Sample data
            if not df.empty:
                description_parts.append("Sample data from the table:")
                sample_rows = min(3, len(df))
                for i in range(sample_rows):
                    row_desc = []
                    for col in df.columns[:5]:  # Limit to first 5 columns
                        value = str(df.iloc[i][col])[:50]  # Limit value length
                        row_desc.append(f"{col}: {value}")
                    description_parts.append(f"Row {i+1}: " + ", ".join(row_desc))
            
            return " ".join(description_parts)
            
        except Exception as e:
            logging.warning(f"Table description generation failed: {e}")
            return f"Table with {len(df)} rows and {len(df.columns)} columns."


class ImageProcessor:
    """Advanced image processing and analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.image_captioning_model = None
        self.object_detection_model = None
        
        # Initialize AI models if available
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models for image understanding."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Image captioning model
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.image_captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logging.info("✅ Image captioning model loaded")
                
                # Object detection
                self.object_detection_model = pipeline("object-detection", model="facebook/detr-resnet-50")
                logging.info("✅ Object detection model loaded")
                
            except Exception as e:
                logging.warning(f"Failed to initialize AI models: {e}")
    
    def process_image(self, image_path: Union[str, Path], image_data: Optional[bytes] = None) -> Optional[MultiModalElement]:
        """Process an image and extract information."""
        if not PIL_AVAILABLE:
            return None
        
        try:
            # Load image
            if image_data:
                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Analyze image
            analysis = self._analyze_image(image)
            
            # Generate description
            description = self._generate_image_description(image, analysis)
            
            # Extract text from image (OCR)
            extracted_text = self._extract_text_from_image(image)
            if extracted_text:
                description += f" Text detected in image: {extracted_text}"
            
            # Detect if it's a chart/diagram
            chart_analysis = self._analyze_chart(image)
            
            element_type = "chart" if chart_analysis.get("is_chart") else "image"
            
            element = MultiModalElement(
                element_id=f"image_{hash(str(image_path)) % 10000}",
                element_type=element_type,
                content=image,
                metadata={
                    "analysis": asdict(analysis),
                    "chart_analysis": chart_analysis,
                    "extraction_method": "pil_analysis",
                    "extracted_text": extracted_text
                },
                text_description=description,
                structured_data=chart_analysis if element_type == "chart" else None,
                confidence_score=0.8,
                processing_method="multimodal_ai"
            )
            
            return element
            
        except Exception as e:
            logging.warning(f"Image processing failed: {e}")
            return None
    
    def _analyze_image(self, image: Image.Image) -> ImageAnalysis:
        """Analyze image properties and content."""
        try:
            # Basic properties
            width, height = image.size
            format_name = image.format or "Unknown"
            color_mode = image.mode
            
            # Dominant colors
            dominant_colors = self._get_dominant_colors(image)
            
            # Generate description using AI model
            description = self._generate_ai_description(image)
            
            # Detect objects
            objects_detected = self._detect_objects(image)
            
            # OCR text extraction
            detected_text = self._extract_text_from_image(image)
            
            return ImageAnalysis(
                width=width,
                height=height,
                format=format_name,
                color_mode=color_mode,
                dominant_colors=dominant_colors,
                detected_text=detected_text,
                description=description,
                objects_detected=objects_detected
            )
            
        except Exception as e:
            logging.warning(f"Image analysis failed: {e}")
            return ImageAnalysis(
                width=0, height=0, format="Unknown", color_mode="Unknown",
                dominant_colors=[], detected_text="", description="Analysis failed",
                objects_detected=[]
            )
    
    def _get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image."""
        try:
            # Resize image for faster processing
            image_small = image.copy()
            image_small.thumbnail((100, 100))
            
            # Convert to numpy array
            pixels = np.array(image_small)
            pixels = pixels.reshape(-1, 3)
            
            # Use k-means clustering to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=min(num_colors, len(pixels)), random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
            
        except Exception as e:
            logging.warning(f"Dominant color extraction failed: {e}")
            return []
    
    def _generate_ai_description(self, image: Image.Image) -> str:
        """Generate AI-powered description of the image."""
        if not self.image_captioning_model:
            return "AI description not available"
        
        try:
            # Generate caption
            inputs = self.processor(image, return_tensors="pt")
            out = self.image_captioning_model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logging.warning(f"AI description generation failed: {e}")
            return "AI description failed"
    
    def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in the image."""
        if not self.object_detection_model:
            return []
        
        try:
            # Run object detection
            results = self.object_detection_model(image)
            
            objects = []
            for result in results:
                objects.append({
                    "label": result["label"],
                    "score": float(result["score"]),
                    "box": result["box"]
                })
            
            return objects
            
        except Exception as e:
            logging.warning(f"Object detection failed: {e}")
            return []
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            # Enhance image for better OCR
            enhanced_image = self._enhance_image_for_ocr(image)
            
            # Extract text
            text = pytesseract.image_to_string(enhanced_image)
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logging.warning(f"OCR failed: {e}")
            return ""
    
    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results."""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Sharpen
            image = image.filter(ImageFilter.SHARPEN)
            
            return image
            
        except Exception as e:
            logging.warning(f"Image enhancement failed: {e}")
            return image
    
    def _analyze_chart(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze if image is a chart and extract chart information."""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Simple heuristics to detect charts
            is_chart = False
            chart_type = None
            chart_data = {}
            
            # Look for chart-like characteristics
            # 1. Check for axis-like structures (lots of horizontal/vertical lines)
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            
            # Edge detection to find lines
            if OPENCV_AVAILABLE:
                edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
                
                # Count horizontal and vertical lines
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
                
                horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
                vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
                
                h_line_count = np.sum(horizontal_lines > 0)
                v_line_count = np.sum(vertical_lines > 0)
                
                # If significant lines detected, likely a chart
                if h_line_count > 100 or v_line_count > 100:
                    is_chart = True
                    
                    # Try to determine chart type based on line patterns
                    if h_line_count > v_line_count * 2:
                        chart_type = "bar_horizontal"
                    elif v_line_count > h_line_count * 2:
                        chart_type = "bar_vertical"
                    elif abs(h_line_count - v_line_count) < min(h_line_count, v_line_count) * 0.5:
                        chart_type = "line_or_scatter"
                    else:
                        chart_type = "mixed"
            
            # Look for text that suggests it's a chart
            detected_text = self._extract_text_from_image(image)
            chart_keywords = ['axis', 'chart', 'graph', 'plot', 'figure', 'data', '%', 'percent']
            
            if any(keyword in detected_text.lower() for keyword in chart_keywords):
                is_chart = True
                if not chart_type:
                    if 'bar' in detected_text.lower():
                        chart_type = "bar"
                    elif 'line' in detected_text.lower():
                        chart_type = "line"
                    elif 'pie' in detected_text.lower():
                        chart_type = "pie"
                    else:
                        chart_type = "unknown"
            
            return {
                "is_chart": is_chart,
                "chart_type": chart_type,
                "detected_text": detected_text,
                "confidence": 0.7 if is_chart else 0.3,
                "analysis_method": "opencv_heuristics" if OPENCV_AVAILABLE else "text_based"
            }
            
        except Exception as e:
            logging.warning(f"Chart analysis failed: {e}")
            return {"is_chart": False, "chart_type": None, "confidence": 0.0}
    
    def _generate_image_description(self, image: Image.Image, analysis: ImageAnalysis) -> str:
        """Generate comprehensive description of the image."""
        try:
            description_parts = []
            
            # Basic properties
            description_parts.append(f"Image dimensions: {analysis.width}x{analysis.height} pixels in {analysis.format} format.")
            
            # AI-generated description
            if analysis.description and analysis.description != "Analysis failed":
                description_parts.append(f"Image shows: {analysis.description}")
            
            # Detected objects
            if analysis.objects_detected:
                objects = [obj["label"] for obj in analysis.objects_detected[:5]]  # Top 5 objects
                description_parts.append(f"Detected objects: {', '.join(objects)}.")
            
            # Color information
            if analysis.dominant_colors:
                color_desc = []
                for r, g, b in analysis.dominant_colors[:3]:  # Top 3 colors
                    color_desc.append(f"RGB({r},{g},{b})")
                description_parts.append(f"Dominant colors: {', '.join(color_desc)}.")
            
            # Text content
            if analysis.detected_text:
                text_preview = analysis.detected_text[:200] + "..." if len(analysis.detected_text) > 200 else analysis.detected_text
                description_parts.append(f"Text content: {text_preview}")
            
            return " ".join(description_parts)
            
        except Exception as e:
            logging.warning(f"Image description generation failed: {e}")
            return f"Image with dimensions {analysis.width}x{analysis.height}."


class MultiModalRAG:
    """Main multi-modal RAG system that coordinates all processors."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.table_processor = TableProcessor(config)
        self.image_processor = ImageProcessor(config)
        
        # Storage for processed elements
        self.multimodal_elements: List[MultiModalElement] = []
        self.element_index: Dict[str, MultiModalElement] = {}
    
    def process_document(self, document: Document) -> List[MultiModalElement]:
        """Process a document and extract multi-modal elements."""
        elements = []
        
        try:
            content = document.page_content
            metadata = document.metadata
            
            # Extract tables from text/HTML
            if '<table' in content.lower() or '|' in content:
                table_elements = self.table_processor.extract_tables_from_html(content)
                if not table_elements:
                    table_elements = self.table_processor.extract_tables_from_text(content)
                elements.extend(table_elements)
            
            # Process images if path information is available
            if 'image_paths' in metadata:
                for image_path in metadata['image_paths']:
                    image_element = self.image_processor.process_image(image_path)
                    if image_element:
                        elements.append(image_element)
            
            # Store elements
            for element in elements:
                self.multimodal_elements.append(element)
                self.element_index[element.element_id] = element
            
            return elements
            
        except Exception as e:
            logging.warning(f"Multi-modal document processing failed: {e}")
            return []
    
    def process_pdf_elements(self, pdf_elements: List[MultiModalElement]) -> List[MultiModalElement]:
        """Process multimodal elements extracted from PDFs with enhanced analysis."""
        processed_elements = []
        
        try:
            for element in pdf_elements:
                if element.element_type == "table":
                    # Enhanced table processing
                    enhanced_table = self._enhance_pdf_table(element)
                    processed_elements.append(enhanced_table)
                    
                elif element.element_type in ["image", "chart"]:
                    # Enhanced image processing
                    enhanced_image = self._enhance_pdf_image(element)
                    processed_elements.append(enhanced_image)
                else:
                    # Keep other elements as-is
                    processed_elements.append(element)
            
            # Store all processed elements
            for element in processed_elements:
                self.multimodal_elements.append(element)
                self.element_index[element.element_id] = element
            
            return processed_elements
            
        except Exception as e:
            logging.warning(f"PDF element processing failed: {e}")
            return pdf_elements
    
    def _enhance_pdf_table(self, table_element: MultiModalElement) -> MultiModalElement:
        """Enhance PDF-extracted table with additional analysis."""
        try:
            if not isinstance(table_element.content, pd.DataFrame):
                return table_element
            
            df = table_element.content
            
            # Additional analysis beyond what was done during extraction
            enhanced_metadata = table_element.metadata.copy()
            
            # Analyze data patterns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Statistical insights
            if numeric_columns:
                stats_summary = {}
                for col in numeric_columns:
                    stats_summary[col] = {
                        'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                        'std': float(df[col].std()) if not df[col].isna().all() else None,
                        'min': float(df[col].min()) if not df[col].isna().all() else None,
                        'max': float(df[col].max()) if not df[col].isna().all() else None
                    }
                enhanced_metadata['statistical_summary'] = stats_summary
            
            # Pattern detection
            patterns = {
                'has_totals': any('total' in str(cell).lower() for cell in df.values.flatten()),
                'has_percentages': any('%' in str(cell) for cell in df.values.flatten()),
                'has_currencies': any('$' in str(cell) or '€' in str(cell) or '£' in str(cell) 
                                    for cell in df.values.flatten()),
                'has_dates': any(self._looks_like_date(str(cell)) for cell in df.values.flatten())
            }
            enhanced_metadata['content_patterns'] = patterns
            
            # Enhanced description
            original_desc = table_element.text_description
            pattern_desc = []
            
            if patterns['has_totals']:
                pattern_desc.append("contains summary totals")
            if patterns['has_percentages']:
                pattern_desc.append("includes percentage values")
            if patterns['has_currencies']:
                pattern_desc.append("contains financial data")
            if patterns['has_dates']:
                pattern_desc.append("includes date information")
            
            if pattern_desc:
                enhanced_desc = f"{original_desc} This table {', '.join(pattern_desc)}."
            else:
                enhanced_desc = original_desc
            
            # Create enhanced element
            enhanced_element = MultiModalElement(
                element_id=table_element.element_id,
                element_type=table_element.element_type,
                content=table_element.content,
                metadata=enhanced_metadata,
                text_description=enhanced_desc,
                structured_data=table_element.structured_data,
                confidence_score=min(1.0, table_element.confidence_score + 0.1),  # Slight boost for enhanced processing
                processing_method=f"{table_element.processing_method}_enhanced"
            )
            
            return enhanced_element
            
        except Exception as e:
            logging.warning(f"Table enhancement failed: {e}")
            return table_element
    
    def _enhance_pdf_image(self, image_element: MultiModalElement) -> MultiModalElement:
        """Enhance PDF-extracted image with additional analysis."""
        try:
            # If we have raw image bytes, process them through our image processor
            if isinstance(image_element.content, bytes):
                # Process the image bytes through our advanced image processor
                processed_image = self.image_processor.process_image(None, image_element.content)
                
                if processed_image:
                    # Merge the analysis results
                    enhanced_metadata = image_element.metadata.copy()
                    enhanced_metadata.update(processed_image.metadata)
                    
                    # Combine descriptions
                    original_desc = image_element.text_description
                    ai_desc = processed_image.text_description
                    combined_desc = f"{original_desc} {ai_desc}" if ai_desc != "Analysis failed" else original_desc
                    
                    # Create enhanced element
                    enhanced_element = MultiModalElement(
                        element_id=image_element.element_id,
                        element_type=processed_image.element_type,  # Might be updated to "chart" if detected
                        content=processed_image.content,  # PIL Image object
                        metadata=enhanced_metadata,
                        text_description=combined_desc,
                        structured_data=processed_image.structured_data,
                        confidence_score=max(image_element.confidence_score, processed_image.confidence_score),
                        processing_method=f"{image_element.processing_method}_ai_enhanced"
                    )
                    
                    return enhanced_element
            
            return image_element
            
        except Exception as e:
            logging.warning(f"Image enhancement failed: {e}")
            return image_element
    
    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date."""
        try:
            import re
            # Simple date patterns
            date_patterns = [
                r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
                r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
                r'\d{4}-\d{1,2}-\d{1,2}',    # YYYY-MM-DD
                r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
            ]
            
            for pattern in date_patterns:
                if re.search(pattern, text):
                    return True
            return False
            
        except Exception:
            return False
    
    def query_multimodal_elements(self, query: str, element_types: Optional[List[str]] = None) -> List[MultiModalElement]:
        """Query multi-modal elements based on text similarity."""
        try:
            if not self.multimodal_elements:
                return []
            
            # Filter by element types if specified
            elements_to_search = self.multimodal_elements
            if element_types:
                elements_to_search = [e for e in self.multimodal_elements if e.element_type in element_types]
            
            # Simple text-based matching for now
            query_lower = query.lower()
            relevant_elements = []
            
            for element in elements_to_search:
                # Check text description
                if query_lower in element.text_description.lower():
                    relevant_elements.append(element)
                    continue
                
                # Check metadata
                metadata_text = json.dumps(element.metadata).lower()
                if query_lower in metadata_text:
                    relevant_elements.append(element)
                    continue
                
                # For tables, check column names and data
                if element.element_type == "table" and element.structured_data:
                    table_text = json.dumps(element.structured_data).lower()
                    if query_lower in table_text:
                        relevant_elements.append(element)
            
            # Sort by confidence score
            relevant_elements.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return relevant_elements
            
        except Exception as e:
            logging.warning(f"Multi-modal query failed: {e}")
            return []
    
    def get_element_summary(self) -> Dict[str, Any]:
        """Get summary of all processed multi-modal elements."""
        try:
            summary = {
                "total_elements": len(self.multimodal_elements),
                "element_types": {},
                "processing_methods": {},
                "average_confidence": 0.0,
                "elements_with_text": 0,
                "elements_with_structured_data": 0
            }
            
            if not self.multimodal_elements:
                return summary
            
            # Count by type
            for element in self.multimodal_elements:
                element_type = element.element_type
                summary["element_types"][element_type] = summary["element_types"].get(element_type, 0) + 1
                
                processing_method = element.processing_method
                summary["processing_methods"][processing_method] = summary["processing_methods"].get(processing_method, 0) + 1
                
                if element.text_description:
                    summary["elements_with_text"] += 1
                
                if element.structured_data:
                    summary["elements_with_structured_data"] += 1
            
            # Average confidence
            if self.multimodal_elements:
                summary["average_confidence"] = sum(e.confidence_score for e in self.multimodal_elements) / len(self.multimodal_elements)
            
            return summary
            
        except Exception as e:
            logging.warning(f"Element summary generation failed: {e}")
            return {"error": str(e)}
    
    def export_elements(self, filepath: str) -> bool:
        """Export all multi-modal elements to a JSON file."""
        try:
            elements_data = []
            
            for element in self.multimodal_elements:
                element_dict = element.to_dict()
                elements_data.append(element_dict)
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total_elements": len(elements_data),
                "summary": self.get_element_summary(),
                "elements": elements_data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logging.error(f"Element export failed: {e}")
            return False
    
    def clear_elements(self):
        """Clear all stored multi-modal elements."""
        self.multimodal_elements.clear()
        self.element_index.clear()
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get current multi-modal processing capabilities."""
        return {
            "table_processing": PANDAS_AVAILABLE,
            "image_processing": PIL_AVAILABLE,
            "advanced_image_processing": OPENCV_AVAILABLE,
            "ocr_text_extraction": TESSERACT_AVAILABLE,
            "ai_image_understanding": TRANSFORMERS_AVAILABLE and self.image_processor.image_captioning_model is not None,
            "object_detection": TRANSFORMERS_AVAILABLE and self.image_processor.object_detection_model is not None,
            "chart_analysis": OPENCV_AVAILABLE and PLOTTING_AVAILABLE
        }