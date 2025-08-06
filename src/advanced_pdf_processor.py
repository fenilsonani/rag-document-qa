"""
Advanced PDF Processing System - Pro-Level Enhancement
Handles tables, images, and complex layouts in multi-page PDFs with precision.
"""

import io
import logging
import tempfile
import base64
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available. Advanced PDF table extraction will be limited.")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("camelot-py not available. Professional table extraction will be limited.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. PDF image extraction will be limited.")

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    logging.warning("tabula-py not available. Java-based table extraction will be limited.")

from langchain.schema import Document
from .config import Config
from .multimodal_rag import MultiModalElement


@dataclass
class PDFLayoutElement:
    """Represents a layout element in a PDF (text block, table, image, etc.)."""
    element_id: str
    element_type: str  # text, table, image, chart, header, footer
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    content: Any
    metadata: Dict[str, Any]
    confidence_score: float = 0.0
    extraction_method: str = ""
    
    def get_area(self) -> float:
        """Calculate the area of the element."""
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0) * (y1 - y0)
    
    def overlaps_with(self, other: 'PDFLayoutElement', threshold: float = 0.1) -> bool:
        """Check if this element overlaps with another."""
        x0, y0, x1, y1 = self.bbox
        ox0, oy0, ox1, oy1 = other.bbox
        
        # Calculate intersection
        ix0 = max(x0, ox0)
        iy0 = max(y0, oy0)
        ix1 = min(x1, ox1)
        iy1 = min(y1, oy1)
        
        if ix0 >= ix1 or iy0 >= iy1:
            return False
        
        intersection_area = (ix1 - ix0) * (iy1 - iy0)
        min_area = min(self.get_area(), other.get_area())
        
        return intersection_area / min_area > threshold


@dataclass
class PDFPageAnalysis:
    """Analysis results for a PDF page."""
    page_number: int
    page_width: float
    page_height: float
    text_blocks: List[PDFLayoutElement]
    tables: List[PDFLayoutElement]
    images: List[PDFLayoutElement]
    layout_columns: int
    reading_order: List[str]  # element_ids in reading order
    

class AdvancedPDFProcessor:
    """Advanced PDF processing with table and image extraction capabilities."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.processed_pages: List[PDFPageAnalysis] = []
        self.extracted_elements: List[PDFLayoutElement] = []
        
    def process_pdf(self, pdf_path: str) -> Tuple[List[Document], List[MultiModalElement]]:
        """Process a PDF file and extract text, tables, and images."""
        try:
            # Convert to Path object
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            documents = []
            multimodal_elements = []
            
            # Extract text with layout information
            text_documents = self._extract_text_with_layout(pdf_path)
            documents.extend(text_documents)
            
            # Extract tables
            table_elements = self._extract_tables_from_pdf(pdf_path)
            multimodal_elements.extend(table_elements)
            
            # Extract images
            image_elements = self._extract_images_from_pdf(pdf_path)
            multimodal_elements.extend(image_elements)
            
            # Analyze page layouts
            self._analyze_page_layouts(pdf_path)
            
            logging.info(f"✅ PDF processing complete: {len(documents)} text documents, "
                        f"{len(table_elements)} tables, {len(image_elements)} images")
            
            return documents, multimodal_elements
            
        except Exception as e:
            logging.error(f"❌ PDF processing failed: {e}")
            return [], []
    
    def _extract_text_with_layout(self, pdf_path: Path) -> List[Document]:
        """Extract text while preserving layout information."""
        documents = []
        
        if not PDFPLUMBER_AVAILABLE:
            # Fallback to basic PyPDF if pdfplumber is not available
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(pdf_path))
            return loader.load()
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with position information
                    text_content = page.extract_text()
                    
                    if text_content:
                        # Get detailed text objects for better layout understanding
                        chars = page.chars
                        
                        # Analyze layout structure
                        layout_info = self._analyze_text_layout(chars, page.width, page.height)
                        
                        document = Document(
                            page_content=text_content,
                            metadata={
                                "source": str(pdf_path),
                                "filename": pdf_path.name,
                                "file_type": ".pdf",
                                "page": page_num + 1,
                                "total_pages": len(pdf.pages),
                                "page_width": page.width,
                                "page_height": page.height,
                                "layout_info": layout_info,
                                "extraction_method": "pdfplumber_layout"
                            }
                        )
                        
                        documents.append(document)
            
            return documents
            
        except Exception as e:
            logging.warning(f"pdfplumber text extraction failed: {e}, falling back to PyPDF")
            # Fallback to basic PyPDF
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(pdf_path))
            return loader.load()
    
    def _analyze_text_layout(self, chars: List[Dict], page_width: float, page_height: float) -> Dict[str, Any]:
        """Analyze text layout to identify columns, headers, etc."""
        if not chars:
            return {"columns": 1, "layout_type": "single_column"}
        
        try:
            # Group characters by line
            lines = {}
            for char in chars:
                y = round(char['y0'], 1)  # Round to handle slight variations
                if y not in lines:
                    lines[y] = []
                lines[y].append(char)
            
            # Analyze column structure
            column_info = self._detect_columns(lines, page_width)
            
            # Detect headers and footers
            sorted_lines = sorted(lines.keys(), reverse=True)  # Top to bottom
            header_lines = sorted_lines[:3] if len(sorted_lines) > 10 else []
            footer_lines = sorted_lines[-3:] if len(sorted_lines) > 10 else []
            
            return {
                "columns": column_info["num_columns"],
                "column_boundaries": column_info["boundaries"],
                "layout_type": column_info["layout_type"],
                "header_lines": header_lines,
                "footer_lines": footer_lines,
                "total_lines": len(lines)
            }
            
        except Exception as e:
            logging.warning(f"Text layout analysis failed: {e}")
            return {"columns": 1, "layout_type": "single_column"}
    
    def _detect_columns(self, lines: Dict[float, List], page_width: float) -> Dict[str, Any]:
        """Detect column structure in the text."""
        try:
            # Analyze x-positions of text starts
            x_positions = []
            for line_chars in lines.values():
                if line_chars:
                    min_x = min(char['x0'] for char in line_chars)
                    max_x = max(char['x1'] for char in line_chars)
                    x_positions.append((min_x, max_x))
            
            if not x_positions:
                return {"num_columns": 1, "boundaries": [], "layout_type": "single_column"}
            
            # Group similar x-start positions
            from collections import Counter
            start_positions = [x[0] for x in x_positions]
            position_counts = Counter([round(pos, -1) for pos in start_positions])  # Round to nearest 10
            
            # If we have 2+ distinct start positions with significant occurrences, likely multi-column
            common_starts = [pos for pos, count in position_counts.items() if count >= 3]
            
            if len(common_starts) >= 2:
                return {
                    "num_columns": len(common_starts),
                    "boundaries": sorted(common_starts),
                    "layout_type": "multi_column"
                }
            else:
                return {"num_columns": 1, "boundaries": [], "layout_type": "single_column"}
                
        except Exception as e:
            logging.warning(f"Column detection failed: {e}")
            return {"num_columns": 1, "boundaries": [], "layout_type": "single_column"}
    
    def _extract_tables_from_pdf(self, pdf_path: Path) -> List[MultiModalElement]:
        """Extract tables from PDF using multiple methods."""
        tables = []
        
        # Method 1: pdfplumber (best for simple tables)
        if PDFPLUMBER_AVAILABLE:
            tables.extend(self._extract_tables_pdfplumber(pdf_path))
        
        # Method 2: camelot (best for complex tables)
        if CAMELOT_AVAILABLE:
            tables.extend(self._extract_tables_camelot(pdf_path))
        
        # Method 3: tabula (Java-based, good for specific formats)
        if TABULA_AVAILABLE:
            tables.extend(self._extract_tables_tabula(pdf_path))
        
        # Remove duplicates based on content similarity
        unique_tables = self._deduplicate_tables(tables)
        
        logging.info(f"📊 Extracted {len(unique_tables)} unique tables from {len(tables)} candidates")
        
        return unique_tables
    
    def _extract_tables_pdfplumber(self, pdf_path: Path) -> List[MultiModalElement]:
        """Extract tables using pdfplumber."""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables from the page
                    page_tables = page.extract_tables()
                    
                    for table_idx, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:  # At least header + 1 row
                            # Convert to DataFrame
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            
                            # Clean the DataFrame
                            df = self._clean_table_dataframe(df)
                            
                            if not df.empty:
                                # Generate description
                                description = self._generate_table_description(df, page_num + 1)
                                
                                element = MultiModalElement(
                                    element_id=f"pdfplumber_table_p{page_num + 1}_t{table_idx}",
                                    element_type="table",
                                    content=df,
                                    metadata={
                                        "page": page_num + 1,
                                        "table_index": table_idx,
                                        "extraction_method": "pdfplumber",
                                        "source_file": str(pdf_path),
                                        "table_bbox": None  # pdfplumber doesn't provide bbox easily
                                    },
                                    text_description=description,
                                    structured_data=df.to_dict(),
                                    confidence_score=0.8,
                                    processing_method="pdfplumber_extraction"
                                )
                                
                                tables.append(element)
            
        except Exception as e:
            logging.warning(f"pdfplumber table extraction failed: {e}")
        
        return tables
    
    def _extract_tables_camelot(self, pdf_path: Path) -> List[MultiModalElement]:
        """Extract tables using camelot-py (more robust for complex tables)."""
        tables = []
        
        try:
            # Extract tables from all pages
            table_list = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')
            
            for table_idx, table in enumerate(table_list):
                if table.df is not None and not table.df.empty:
                    df = table.df.copy()
                    
                    # Clean the DataFrame
                    df = self._clean_table_dataframe(df)
                    
                    if not df.empty:
                        # Generate description
                        page_num = table.page if hasattr(table, 'page') else 1
                        description = self._generate_table_description(df, page_num)
                        
                        element = MultiModalElement(
                            element_id=f"camelot_table_{table_idx}",
                            element_type="table",
                            content=df,
                            metadata={
                                "page": page_num,
                                "table_index": table_idx,
                                "extraction_method": "camelot",
                                "source_file": str(pdf_path),
                                "accuracy": table.accuracy if hasattr(table, 'accuracy') else None,
                                "whitespace": table.whitespace if hasattr(table, 'whitespace') else None,
                                "table_bbox": table._bbox if hasattr(table, '_bbox') else None
                            },
                            text_description=description,
                            structured_data=df.to_dict(),
                            confidence_score=min(0.9, table.accuracy / 100) if hasattr(table, 'accuracy') else 0.7,
                            processing_method="camelot_lattice"
                        )
                        
                        tables.append(element)
            
            # Try stream flavor if lattice didn't find many tables
            if len(tables) < 2:
                try:
                    stream_tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='stream')
                    
                    for table_idx, table in enumerate(stream_tables):
                        if table.df is not None and not table.df.empty:
                            df = self._clean_table_dataframe(table.df.copy())
                            
                            if not df.empty:
                                page_num = table.page if hasattr(table, 'page') else 1
                                description = self._generate_table_description(df, page_num)
                                
                                element = MultiModalElement(
                                    element_id=f"camelot_stream_table_{table_idx}",
                                    element_type="table",
                                    content=df,
                                    metadata={
                                        "page": page_num,
                                        "table_index": table_idx,
                                        "extraction_method": "camelot_stream",
                                        "source_file": str(pdf_path),
                                        "accuracy": table.accuracy if hasattr(table, 'accuracy') else None
                                    },
                                    text_description=description,
                                    structured_data=df.to_dict(),
                                    confidence_score=0.6,
                                    processing_method="camelot_stream"
                                )
                                
                                tables.append(element)
                
                except Exception as stream_error:
                    logging.warning(f"Camelot stream extraction failed: {stream_error}")
            
        except Exception as e:
            logging.warning(f"camelot table extraction failed: {e}")
        
        return tables
    
    def _extract_tables_tabula(self, pdf_path: Path) -> List[MultiModalElement]:
        """Extract tables using tabula-py (Java-based)."""
        tables = []
        
        try:
            # Extract tables from all pages
            dfs = tabula.read_pdf(str(pdf_path), pages='all', multiple_tables=True)
            
            for table_idx, df in enumerate(dfs):
                if df is not None and not df.empty:
                    # Clean the DataFrame
                    df = self._clean_table_dataframe(df)
                    
                    if not df.empty:
                        # Generate description (page number not easily available from tabula)
                        description = self._generate_table_description(df, "unknown")
                        
                        element = MultiModalElement(
                            element_id=f"tabula_table_{table_idx}",
                            element_type="table",
                            content=df,
                            metadata={
                                "table_index": table_idx,
                                "extraction_method": "tabula",
                                "source_file": str(pdf_path)
                            },
                            text_description=description,
                            structured_data=df.to_dict(),
                            confidence_score=0.6,
                            processing_method="tabula_java"
                        )
                        
                        tables.append(element)
            
        except Exception as e:
            logging.warning(f"tabula table extraction failed: {e}")
        
        return tables
    
    def _clean_table_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize a table DataFrame."""
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Clean column names
            df.columns = [str(col).strip() if col is not None else f"Column_{i}" 
                         for i, col in enumerate(df.columns)]
            
            # Remove duplicate column names
            seen = set()
            new_columns = []
            for col in df.columns:
                if col in seen:
                    counter = 1
                    new_col = f"{col}_{counter}"
                    while new_col in seen:
                        counter += 1
                        new_col = f"{col}_{counter}"
                    new_columns.append(new_col)
                    seen.add(new_col)
                else:
                    new_columns.append(col)
                    seen.add(col)
            
            df.columns = new_columns
            
            # Clean cell values
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # Replace common problematic values
                df[col] = df[col].replace(['nan', 'None', 'null', ''], pd.NA)
            
            # Remove rows that are mostly empty
            min_non_na = max(1, len(df.columns) // 2)  # At least half the columns should have data
            df = df.dropna(thresh=min_non_na)
            
            return df
            
        except Exception as e:
            logging.warning(f"Table cleaning failed: {e}")
            return df
    
    def _generate_table_description(self, df: pd.DataFrame, page_num: Union[int, str]) -> str:
        """Generate a description for an extracted table."""
        try:
            description_parts = []
            
            # Basic info
            page_info = f" on page {page_num}" if page_num != "unknown" else ""
            description_parts.append(f"Table{page_info} with {len(df)} rows and {len(df.columns)} columns.")
            
            # Column information
            description_parts.append(f"Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}.")
            
            # Sample data
            if not df.empty:
                description_parts.append("Sample data:")
                for i in range(min(2, len(df))):
                    row_data = []
                    for col in df.columns[:3]:  # First 3 columns
                        value = str(df.iloc[i][col])[:30]  # Limit length
                        row_data.append(f"{col}: {value}")
                    description_parts.append(f"Row {i+1}: {', '.join(row_data)}")
            
            return " ".join(description_parts)
            
        except Exception as e:
            logging.warning(f"Table description generation failed: {e}")
            return f"Table with {len(df)} rows and {len(df.columns)} columns."
    
    def _deduplicate_tables(self, tables: List[MultiModalElement]) -> List[MultiModalElement]:
        """Remove duplicate tables based on content similarity."""
        if len(tables) <= 1:
            return tables
        
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            
            for existing_table in unique_tables:
                if self._tables_are_similar(table, existing_table):
                    # Choose the one with higher confidence
                    if table.confidence_score > existing_table.confidence_score:
                        # Replace the existing table
                        unique_tables.remove(existing_table)
                        unique_tables.append(table)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _tables_are_similar(self, table1: MultiModalElement, table2: MultiModalElement, threshold: float = 0.8) -> bool:
        """Check if two tables are similar enough to be considered duplicates."""
        try:
            if not isinstance(table1.content, pd.DataFrame) or not isinstance(table2.content, pd.DataFrame):
                return False
            
            df1, df2 = table1.content, table2.content
            
            # Basic shape similarity
            if abs(len(df1) - len(df2)) > max(len(df1), len(df2)) * 0.2:  # More than 20% difference in rows
                return False
            
            if abs(len(df1.columns) - len(df2.columns)) > 1:  # More than 1 column difference
                return False
            
            # Content similarity (simple string comparison)
            str1 = df1.to_string()
            str2 = df2.to_string()
            
            # Calculate simple similarity
            shorter_len = min(len(str1), len(str2))
            if shorter_len == 0:
                return False
            
            common_chars = sum(1 for a, b in zip(str1, str2) if a == b)
            similarity = common_chars / shorter_len
            
            return similarity > threshold
            
        except Exception as e:
            logging.warning(f"Table similarity check failed: {e}")
            return False
    
    def _extract_images_from_pdf(self, pdf_path: Path) -> List[MultiModalElement]:
        """Extract images from PDF using PyMuPDF."""
        images = []
        
        if not PYMUPDF_AVAILABLE:
            logging.warning("PyMuPDF not available, skipping image extraction")
            return images
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to PIL Image format
                        img_data = None
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        pix = None
                        
                        if img_data:
                            # Get image bbox on page
                            img_rects = page.get_image_rects(img)
                            bbox = img_rects[0] if img_rects else (0, 0, 100, 100)
                            
                            # Create MultiModalElement for image processing
                            element = MultiModalElement(
                                element_id=f"pdf_image_p{page_num + 1}_i{img_index}",
                                element_type="image",
                                content=img_data,  # Raw image bytes
                                metadata={
                                    "page": page_num + 1,
                                    "image_index": img_index,
                                    "extraction_method": "pymupdf",
                                    "source_file": str(pdf_path),
                                    "bbox": bbox,
                                    "xref": xref
                                },
                                text_description=f"Image extracted from page {page_num + 1}",
                                structured_data=None,
                                confidence_score=0.9,
                                processing_method="pymupdf_extraction"
                            )
                            
                            images.append(element)
                    
                    except Exception as img_error:
                        logging.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {img_error}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logging.warning(f"PyMuPDF image extraction failed: {e}")
        
        return images
    
    def _analyze_page_layouts(self, pdf_path: Path) -> None:
        """Analyze the layout structure of PDF pages."""
        if not PDFPLUMBER_AVAILABLE:
            return
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Create page analysis
                    analysis = PDFPageAnalysis(
                        page_number=page_num + 1,
                        page_width=page.width,
                        page_height=page.height,
                        text_blocks=[],
                        tables=[],
                        images=[],
                        layout_columns=1,
                        reading_order=[]
                    )
                    
                    # Analyze text layout
                    chars = page.chars
                    if chars:
                        layout_info = self._analyze_text_layout(chars, page.width, page.height)
                        analysis.layout_columns = layout_info["columns"]
                    
                    self.processed_pages.append(analysis)
        
        except Exception as e:
            logging.warning(f"Page layout analysis failed: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the PDF processing results."""
        return {
            "total_pages_analyzed": len(self.processed_pages),
            "total_elements_extracted": len(self.extracted_elements),
            "elements_by_type": {
                elem_type: len([e for e in self.extracted_elements if e.element_type == elem_type])
                for elem_type in set(e.element_type for e in self.extracted_elements)
            } if self.extracted_elements else {},
            "processing_methods_used": [
                "pdfplumber" if PDFPLUMBER_AVAILABLE else None,
                "camelot" if CAMELOT_AVAILABLE else None,
                "tabula" if TABULA_AVAILABLE else None,
                "pymupdf" if PYMUPDF_AVAILABLE else None
            ],
            "capabilities": {
                "advanced_table_extraction": CAMELOT_AVAILABLE,
                "layout_aware_text": PDFPLUMBER_AVAILABLE,
                "image_extraction": PYMUPDF_AVAILABLE,
                "java_table_extraction": TABULA_AVAILABLE
            }
        }
    
    def clear_cache(self):
        """Clear processed data to free memory."""
        self.processed_pages.clear()
        self.extracted_elements.clear()