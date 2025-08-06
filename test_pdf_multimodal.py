#!/usr/bin/env python3
"""
Advanced PDF Processing Demo - Test Script
Demonstrates table and image extraction from multi-page PDFs
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.advanced_pdf_processor import AdvancedPDFProcessor
    from src.document_loader import DocumentProcessor
    from src.multimodal_rag import MultiModalRAG
    from src.enhanced_rag import EnhancedRAG
    from src.config import Config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def create_sample_pdf_with_table_and_image():
    """Create a sample PDF with tables and images for testing."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        import matplotlib.pyplot as plt
        import pandas as pd
        import io
        
        # Create sample data
        data = {
            'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
            'Price': [999.99, 29.99, 79.99, 299.99, 149.99],
            'Stock': [15, 50, 30, 8, 25],
            'Rating': [4.5, 4.2, 4.7, 4.3, 4.6]
        }
        df = pd.DataFrame(data)
        
        # Create a sample chart
        plt.figure(figsize=(6, 4))
        plt.bar(df['Product'], df['Price'])
        plt.title('Product Prices')
        plt.xlabel('Products')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart to bytes
        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
        chart_buffer.seek(0)
        plt.close()
        
        # Create temporary PDF file
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        
        # Create PDF document
        doc = SimpleDocTemplate(temp_pdf.name, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Sample Multi-Modal PDF Document", title_style))
        story.append(Spacer(1, 20))
        
        # Introduction paragraph
        intro_text = """
        This is a sample PDF document created for testing advanced PDF processing capabilities.
        It contains both tabular data and charts to demonstrate multi-modal content extraction.
        The system should be able to extract and process both the table and image content.
        """
        story.append(Paragraph(intro_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Create table data for PDF
        table_data = [['Product', 'Price ($)', 'Stock', 'Rating']]
        for _, row in df.iterrows():
            table_data.append([row['Product'], f"${row['Price']:.2f}", str(row['Stock']), str(row['Rating'])])
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Product Inventory Table", styles['Heading2']))
        story.append(Spacer(1, 10))
        story.append(table)
        story.append(Spacer(1, 30))
        
        # Add the chart
        story.append(Paragraph("Product Price Chart", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Create image from buffer
        chart_image = Image(chart_buffer, width=5*inch, height=3*inch)
        story.append(chart_image)
        story.append(Spacer(1, 20))
        
        # Add more content for second page
        story.append(Paragraph("Additional Analysis", styles['Heading2']))
        analysis_text = """
        The inventory data shows varying stock levels across products. Laptops have the highest price point
        but relatively low stock (15 units), while mice are the most affordable with high availability (50 units).
        All products maintain good customer ratings above 4.0, indicating quality satisfaction.
        
        This multi-page document demonstrates the system's ability to:
        • Extract structured tables with multiple data types
        • Process embedded images and charts
        • Maintain layout awareness across pages
        • Handle mixed content types in a single document
        """
        story.append(Paragraph(analysis_text, styles['Normal']))
        
        # Build the PDF
        doc.build(story)
        temp_pdf.close()
        
        return temp_pdf.name
        
    except ImportError as e:
        print(f"⚠️ Could not create sample PDF: {e}")
        print("Install reportlab and matplotlib: pip install reportlab matplotlib")
        return None
    except Exception as e:
        print(f"❌ Error creating sample PDF: {e}")
        return None


def test_advanced_pdf_processor():
    """Test the AdvancedPDFProcessor directly."""
    print("🔧 Testing AdvancedPDFProcessor...")
    
    # Create sample PDF
    pdf_path = create_sample_pdf_with_table_and_image()
    if not pdf_path:
        print("❌ Could not create sample PDF for testing")
        return False
    
    try:
        processor = AdvancedPDFProcessor()
        print(f"📄 Processing PDF: {pdf_path}")
        
        # Process the PDF
        documents, multimodal_elements = processor.process_pdf(pdf_path)
        
        print(f"✅ Extracted {len(documents)} text documents")
        print(f"✅ Extracted {len(multimodal_elements)} multimodal elements")
        
        # Show multimodal elements
        tables = [e for e in multimodal_elements if e.element_type == "table"]
        images = [e for e in multimodal_elements if e.element_type in ["image", "chart"]]
        
        print(f"📊 Tables found: {len(tables)}")
        for i, table in enumerate(tables):
            print(f"   Table {i+1}: {table.processing_method} (confidence: {table.confidence_score:.2f})")
            if hasattr(table.content, 'shape'):
                print(f"   Shape: {table.content.shape}")
            print(f"   Description: {table.text_description[:100]}...")
        
        print(f"🖼️ Images found: {len(images)}")
        for i, image in enumerate(images):
            print(f"   Image {i+1}: {image.processing_method} (confidence: {image.confidence_score:.2f})")
            print(f"   Description: {image.text_description[:100]}...")
        
        # Get processing summary
        summary = processor.get_processing_summary()
        print(f"📈 Processing Summary:")
        print(f"   Pages analyzed: {summary['total_pages_analyzed']}")
        print(f"   Elements extracted: {summary['total_elements_extracted']}")
        print(f"   Capabilities: {summary['capabilities']}")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedPDFProcessor test failed: {e}")
        return False
    
    finally:
        # Clean up
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_document_processor_integration():
    """Test the integration with DocumentProcessor."""
    print("\n🔧 Testing DocumentProcessor integration...")
    
    # Create sample PDF
    pdf_path = create_sample_pdf_with_table_and_image()
    if not pdf_path:
        print("❌ Could not create sample PDF for testing")
        return False
    
    try:
        processor = DocumentProcessor()
        print(f"📄 Processing PDF through DocumentProcessor: {pdf_path}")
        
        # Process the document
        documents = processor.load_document(pdf_path)
        
        print(f"✅ Loaded {len(documents)} document chunks")
        
        # Check multimodal elements
        multimodal_elements = processor.get_multimodal_elements()
        tables = processor.get_tables()
        images = processor.get_images()
        
        print(f"📊 Multimodal elements: {len(multimodal_elements)}")
        print(f"📊 Tables: {len(tables)}")
        print(f"🖼️ Images: {len(images)}")
        
        # Get processing summary
        summary = processor.get_processing_summary()
        print(f"📈 Processing Summary: {json.dumps(summary, indent=2, default=str)}")
        
        return True
        
    except Exception as e:
        print(f"❌ DocumentProcessor integration test failed: {e}")
        return False
    
    finally:
        # Clean up
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_enhanced_rag_system():
    """Test the full EnhancedRAG system with PDF processing."""
    print("\n🔧 Testing EnhancedRAG system...")
    
    # Create sample PDF
    pdf_path = create_sample_pdf_with_table_and_image()
    if not pdf_path:
        print("❌ Could not create sample PDF for testing")
        return False
    
    try:
        # Initialize enhanced RAG
        rag = EnhancedRAG()
        
        # Initialize system
        print("🚀 Initializing Enhanced RAG system...")
        init_result = rag.initialize()
        print(f"✅ System initialized: {init_result['success']}")
        print(f"Mode: {init_result.get('mode', 'unknown')}")
        
        # Process documents
        print(f"📄 Processing documents...")
        process_result = rag.process_documents([pdf_path])
        
        if process_result["success"]:
            print(f"✅ Documents processed successfully")
            print(f"📊 Processed files: {process_result.get('processed_files', 0)}")
            print(f"📝 Total chunks: {process_result.get('total_chunks', 0)}")
            
            # Check multimodal processing
            if "multimodal_processing" in process_result:
                mm_info = process_result["multimodal_processing"]
                print(f"🎭 Multimodal processing enabled: {mm_info['enabled']}")
                print(f"📊 Elements extracted: {mm_info['elements_extracted']}")
                print(f"🔧 Element types: {mm_info['element_types']}")
        
        # Get multimodal elements from RAG system
        tables = rag.get_tables()
        images = rag.get_images()
        
        print(f"📊 Tables accessible: {len(tables)}")
        print(f"🖼️ Images accessible: {len(images)}")
        
        # Get comprehensive summary
        summary = rag.get_processing_summary()
        print(f"📈 Enhanced RAG Summary:")
        print(f"   System mode: {summary['system_mode']}")
        print(f"   Documents processed: {summary['documents_processed']}")
        print(f"   Multimodal elements: {summary['multimodal_elements']}")
        print(f"   Features enabled: {summary['features_enabled']}")
        
        # Test multimodal search
        if tables or images:
            print("\n🔍 Testing multimodal search...")
            search_result = rag.search_multimodal_content("price table product inventory")
            print(f"Search results: {search_result.get('total_found', 0)} found")
            
            for result in search_result.get('results', [])[:2]:  # Show first 2 results
                print(f"   - {result['element_type']}: {result['description'][:80]}...")
        
        # Test a question
        if process_result["success"]:
            print("\n❓ Testing question answering...")
            qa_result = rag.ask_question("What products are in the inventory table and what are their prices?")
            
            if qa_result.get("success"):
                print(f"✅ Question answered successfully")
                print(f"Answer: {qa_result.get('answer', 'No answer')[:200]}...")
            else:
                print(f"⚠️ Question answering failed: {qa_result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ EnhancedRAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)


def main():
    """Run all tests."""
    print("🧪 Advanced PDF Processing Test Suite")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    try:
        import pdfplumber
        print("✅ pdfplumber available")
    except ImportError:
        print("❌ pdfplumber not available - install with: pip install pdfplumber")
    
    try:
        import camelot
        print("✅ camelot available")
    except ImportError:
        print("⚠️ camelot not available - install with: pip install camelot-py[cv]")
    
    try:
        import fitz
        print("✅ PyMuPDF available")
    except ImportError:
        print("⚠️ PyMuPDF not available - install with: pip install PyMuPDF")
    
    try:
        import tabula
        print("✅ tabula available")
    except ImportError:
        print("⚠️ tabula not available - install with: pip install tabula-py")
    
    print("\n" + "=" * 50)
    
    # Run tests
    test_results = []
    
    # Test 1: Advanced PDF Processor
    test_results.append(("AdvancedPDFProcessor", test_advanced_pdf_processor()))
    
    # Test 2: Document Processor Integration
    test_results.append(("DocumentProcessor Integration", test_document_processor_integration()))
    
    # Test 3: Enhanced RAG System
    test_results.append(("Enhanced RAG System", test_enhanced_rag_system()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your PDF processing system is ready.")
        print("\n📋 Next steps:")
        print("1. Install any missing dependencies shown above")
        print("2. Test with your own PDF files containing tables and images")
        print("3. Use the enhanced RAG system through the main Streamlit app")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("Make sure all dependencies are installed correctly.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)