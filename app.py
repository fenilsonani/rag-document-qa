"""
RAG Document Q&A System
A Streamlit application for document-based question answering using LangChain and RAG.
"""

import streamlit as st
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

from src.enhanced_rag import EnhancedRAG
from src.config import Config
from src.model_manager import ModelManager, ModelDownloadProgress

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "enhanced_rag" not in st.session_state:
    st.session_state.enhanced_rag = EnhancedRAG()
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.system_mode = "offline"  # Will be set during initialization

if "config" not in st.session_state:
    st.session_state.config = Config()

if "document_insights" not in st.session_state:
    st.session_state.document_insights = None

if "relationship_analysis" not in st.session_state:
    st.session_state.relationship_analysis = None

if "smart_suggestions" not in st.session_state:
    st.session_state.smart_suggestions = []

if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager()

if "download_progress" not in st.session_state:
    st.session_state.download_progress = {}

def initialize_system(force_mode=None) -> bool:
    """Initialize the enhanced RAG system."""
    if not st.session_state.initialized or force_mode:
        with st.spinner("🚀 Initializing Enhanced RAG System..."):
            result = st.session_state.enhanced_rag.initialize(force_mode=force_mode)
            
            if result["success"]:
                st.session_state.initialized = True
                st.session_state.system_mode = result["mode"]
                
                # Show mode-specific success message
                if result["mode"] == "online":
                    st.success("✅ Online mode enabled with API keys!")
                elif result["mode"] == "offline":
                    st.success("🔧 Offline mode enabled (no API keys needed)!")
                else:
                    st.info("⚠️ Basic mode (limited functionality)")
                
                # Show initialization details
                with st.expander("📋 System Initialization Details"):
                    st.json(result)
                
                return True
            else:
                st.error(f"❌ System initialization failed")
                return False
    
    return True

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to uploads directory."""
    upload_dir = Path(st.session_state.config.UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def display_sources(sources: List[Dict[str, Any]]) -> None:
    """Display source documents in an organized manner."""
    if not sources:
        return
    
    st.subheader("📄 Sources")
    
    for source in sources:
        with st.expander(f"Source {source['index']}: {source['filename']}"):
            st.write("**Content Preview:**")
            st.write(source['content'])
            
            st.write("**Metadata:**")
            metadata = source['metadata']
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**File:** {metadata.get('filename', 'Unknown')}")
                st.write(f"**File Type:** {metadata.get('file_type', 'Unknown')}")
            
            with col2:
                st.write(f"**Chunk ID:** {metadata.get('chunk_id', 'Unknown')}")
                st.write(f"**Chunk Size:** {metadata.get('chunk_size', 'Unknown')} chars")

def main():
    """Main application function."""
    # Header
    st.title("🚀 Pro-Level RAG Document Q&A System")
    st.markdown("**Professional-grade document intelligence** with hybrid search, cross-encoder reranking, and comprehensive evaluation framework.")
    
    # Show pro-level features banner
    status = st.session_state.enhanced_rag.get_system_status()
    capabilities = status.get('capabilities', {})
    
    feature_badges = []
    if capabilities.get('hybrid_search'):
        feature_badges.append("🎯 **Hybrid Search**")
    if capabilities.get('reranking'):
        feature_badges.append("🎭 **Cross-Encoder Reranking**")
    if capabilities.get('comprehensive_evaluation'):
        feature_badges.append("📊 **Evaluation Suite**")
    if capabilities.get('document_intelligence'):
        feature_badges.append("🧠 **Document Intelligence**")
    
    if feature_badges:
        st.info(f"**Pro Features Active:** {' • '.join(feature_badges)}")
    else:
        st.warning("⚡ **Basic Mode** - Upload documents to unlock pro-level features!")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # System Mode and Status
        st.subheader("🚀 System Mode")
        
        mode_icons = {
            "online": "🌐",
            "offline": "🔧", 
            "basic": "⚠️",
            "auto": "🤖"
        }
        
        # Mode selector
        api_keys_available = bool(st.session_state.config.OPENAI_API_KEY) or bool(st.session_state.config.ANTHROPIC_API_KEY)
        
        mode_options = ["🤖 Auto (Smart Choice)", "🔧 Offline Mode (Local Models)"]
        if api_keys_available:
            mode_options.insert(1, "🌐 Online Mode (API Keys)")
        
        # Map display names to internal values
        mode_mapping = {
            "🤖 Auto (Smart Choice)": None,
            "🌐 Online Mode (API Keys)": "online", 
            "🔧 Offline Mode (Local Models)": "offline"
        }
        
        current_mode = st.session_state.get('system_mode', 'offline')
        default_index = 0  # Default to Auto
        
        # Find current selection based on mode
        if current_mode == "online" and api_keys_available:
            default_index = 1 if api_keys_available else 0
        elif current_mode == "offline":
            default_index = len(mode_options) - 1
        
        selected_mode_display = st.selectbox(
            "Choose operating mode:",
            mode_options,
            index=default_index,
            help="• **Auto**: Automatically chooses the best available mode\n• **Online**: Uses API keys for enhanced features\n• **Offline**: Uses local models, no API keys needed"
        )
        
        selected_mode = mode_mapping[selected_mode_display]
        
        # Apply mode change if different
        if st.button("🔄 Apply Mode", help="Click to switch to the selected mode"):
            with st.spinner(f"Switching to {selected_mode_display}..."):
                # Reset system for mode change
                from src.enhanced_rag import EnhancedRAG
                st.session_state.enhanced_rag = EnhancedRAG()
                st.session_state.initialized = False
                
                # Initialize with selected mode
                if initialize_system(force_mode=selected_mode):
                    st.success(f"✅ Switched to {selected_mode_display}!")
                    st.rerun()
                else:
                    st.error("❌ Failed to switch mode")
        
        # Display current active mode
        current_mode = st.session_state.get('system_mode', 'offline')
        st.write(f"**Active:** {mode_icons.get(current_mode, '❓')} {current_mode.title()} Mode")
        
        # API Key Status
        api_keys_status = {
            "OpenAI": bool(st.session_state.config.OPENAI_API_KEY),
            "Anthropic": bool(st.session_state.config.ANTHROPIC_API_KEY)
        }
        
        st.subheader("🔑 API Keys Status")
        for provider, status in api_keys_status.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {provider}")
        
        if not any(api_keys_status.values()):
            st.info("💡 **No API keys needed!** System runs in offline mode with local models.")
        else:
            st.success("🚀 API keys available - enhanced online features enabled!")
        
        # System Status
        if st.session_state.initialized:
            st.subheader("📊 System Status")
            status = st.session_state.enhanced_rag.get_system_status()
            
            mode = status.get('mode', 'unknown')
            st.write(f"🤖 Mode: {mode.title()}")
            st.write(f"📄 Documents: {status.get('documents_processed', 0)}")
            st.write(f"💬 Questions: {len(st.session_state.chat_history)}")
            
            capabilities = status.get('capabilities', {})
            qa_status = "✅" if capabilities.get('question_answering') else "❌"
            st.write(f"🔗 Q&A: {qa_status}")
            
            intel_status = "✅" if capabilities.get('document_intelligence') else "❌"  
            st.write(f"🧠 Intelligence: {intel_status}")
        
        # Settings
        st.subheader("⚙️ Settings")
        use_conversation = st.checkbox("Enable Conversation Mode", value=True)
        
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.success("Conversation cleared!")
        
        # Model Management for Offline Mode
        if current_mode == "offline":
            st.subheader("🤖 Offline Models")
            
            # Get model manager and current status
            model_manager = st.session_state.model_manager
            
            # Storage info
            storage_info = model_manager.get_storage_info()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📥 Downloaded", f"{storage_info['downloaded_models']}")
            with col2:
                st.metric("💾 Storage", f"{storage_info['total_size_mb']:.0f}MB")
            
            # Model selection with radio buttons
            st.write("**Question Answering Models:**")
            qa_models = model_manager.get_models_by_task("question-answering")
            qa_options = []
            qa_keys = []
            
            for key, model_info in qa_models.items():
                status_icon = "✅" if model_info.is_downloaded else "📥"
                size_text = f"({model_info.size_mb}MB)"
                performance = f"★{model_info.performance_score:.1f}"
                option_text = f"{status_icon} {model_info.name} {size_text} {performance}"
                qa_options.append(option_text)
                qa_keys.append(key)
            
            if qa_options:
                # Get current selection index
                current_qa_model = st.session_state.enhanced_rag.offline_rag.selected_models.get("qa_model", "qa_distilbert")
                current_qa_index = qa_keys.index(current_qa_model) if current_qa_model in qa_keys else 0
                
                selected_qa_index = st.radio(
                    "Select QA Model:",
                    range(len(qa_options)),
                    format_func=lambda x: qa_options[x],
                    index=current_qa_index,
                    key="qa_model_selection"
                )
                selected_qa_key = qa_keys[selected_qa_index]
            
            st.write("**Summarization Models:**")
            summarizer_models = model_manager.get_models_by_task("summarization")
            sum_options = []
            sum_keys = []
            
            for key, model_info in summarizer_models.items():
                status_icon = "✅" if model_info.is_downloaded else "📥"
                size_text = f"({model_info.size_mb}MB)"
                performance = f"★{model_info.performance_score:.1f}"
                option_text = f"{status_icon} {model_info.name} {size_text} {performance}"
                sum_options.append(option_text)
                sum_keys.append(key)
            
            if sum_options:
                # Get current selection index
                current_sum_model = st.session_state.enhanced_rag.offline_rag.selected_models.get("summarizer_model", "summarizer_t5")
                current_sum_index = sum_keys.index(current_sum_model) if current_sum_model in sum_keys else 0
                
                selected_sum_index = st.radio(
                    "Select Summarizer:",
                    range(len(sum_options)),
                    format_func=lambda x: sum_options[x],
                    index=current_sum_index,
                    key="summarizer_model_selection"
                )
                selected_sum_key = sum_keys[selected_sum_index]
            
            # Download required models
            models_to_download = []
            selected_models = {}
            
            if 'selected_qa_key' in locals():
                selected_models["qa_model"] = selected_qa_key
                if not model_manager.is_model_downloaded(selected_qa_key):
                    models_to_download.append(selected_qa_key)
            
            if 'selected_sum_key' in locals():
                selected_models["summarizer_model"] = selected_sum_key
                if not model_manager.is_model_downloaded(selected_sum_key):
                    models_to_download.append(selected_sum_key)
            
            # Download button and progress
            if models_to_download:
                total_size = model_manager.estimate_total_download_size(models_to_download)
                
                if st.button(f"📥 Download Models ({total_size}MB)", type="primary"):
                    st.session_state.downloading_models = True
                    st.session_state.models_to_download = models_to_download
                    st.session_state.selected_models = selected_models
                    st.rerun()
            
            # Handle model downloading with progress
            if st.session_state.get('downloading_models', False):
                models_to_download = st.session_state.get('models_to_download', [])
                selected_models = st.session_state.get('selected_models', {})
                
                st.write("**🔄 Downloading Models...**")
                
                # Create progress containers
                progress_containers = {}
                for model_key in models_to_download:
                    model_info = model_manager.get_model_info(model_key) 
                    if model_info:
                        with st.container():
                            st.write(f"**{model_info.name}**")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            progress_containers[model_key] = {
                                "progress_bar": progress_bar,
                                "status_text": status_text,
                                "model_info": model_info
                            }
                
                # Download models with progress tracking
                def update_progress(model_key: str, progress: ModelDownloadProgress):
                    if model_key in progress_containers:
                        container = progress_containers[model_key]
                        percentage = progress.get_progress_percentage()
                        container["progress_bar"].progress(percentage / 100.0)
                        
                        if progress.status == "downloading":
                            speed = progress.get_speed_mbps()
                            eta = progress.get_eta_seconds()
                            status_msg = f"Downloading... {percentage:.1f}% ({speed:.1f} MB/s, ETA: {eta:.0f}s)"
                        elif progress.status == "completed":
                            status_msg = "✅ Download completed!"
                        elif progress.status == "error":
                            status_msg = f"❌ Error: {progress.error_message}"
                        else:
                            status_msg = f"Status: {progress.status}"
                        
                        container["status_text"].write(status_msg)
                
                try:
                    # Download models
                    results = model_manager.download_multiple_models(
                        models_to_download,
                        update_progress
                    )
                    
                    if results["success"]:
                        st.success(f"✅ Successfully downloaded {results['total_downloaded']} models!")
                        
                        # Update offline RAG with new models
                        st.session_state.enhanced_rag.offline_rag.set_selected_models(selected_models)
                        
                        # Clear downloading state
                        st.session_state.downloading_models = False
                        st.session_state.models_to_download = []
                        st.session_state.selected_models = {}
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to download some models: {results['failed_models']}")
                        st.session_state.downloading_models = False
                
                except Exception as e:
                    st.error(f"❌ Download error: {e}")
                    st.session_state.downloading_models = False
            
            # Model presets
            with st.expander("🎯 Model Presets"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("⚡ Lightweight", help="Fast models for limited resources"):
                        preset_models = model_manager.get_lightweight_models()
                        st.session_state.preset_models = preset_models
                        st.rerun()
                
                with col2:
                    if st.button("🎯 Recommended", help="Balanced performance and speed"):
                        preset_models = model_manager.get_recommended_models()
                        st.session_state.preset_models = preset_models
                        st.rerun()
                
                with col3:
                    if st.button("🚀 High Performance", help="Best accuracy models"):
                        preset_models = model_manager.get_high_performance_models()
                        st.session_state.preset_models = preset_models
                        st.rerun()
            
            # Handle preset selection
            if st.session_state.get('preset_models'):
                preset_models = st.session_state.preset_models
                total_size = model_manager.estimate_total_download_size(preset_models)
                
                st.info(f"Selected preset requires {total_size}MB download")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Apply Preset"):
                        # Download and apply preset
                        st.session_state.downloading_models = True
                        st.session_state.models_to_download = [m for m in preset_models if not model_manager.is_model_downloaded(m)]
                        st.session_state.selected_models = {
                            "qa_model": preset_models[0] if len(preset_models) > 0 else "qa_distilbert",
                            "summarizer_model": preset_models[1] if len(preset_models) > 1 else "summarizer_t5"
                        }
                        st.session_state.preset_models = None
                        st.rerun()
                
                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state.preset_models = None
                        st.rerun()
        
        if st.button("🔄 Reset System"):
            from src.enhanced_rag import EnhancedRAG
            st.session_state.enhanced_rag = EnhancedRAG()
            st.session_state.initialized = False
            st.session_state.chat_history = []
            st.session_state.document_insights = None
            st.session_state.relationship_analysis = None
            st.session_state.smart_suggestions = []
            st.success("System reset!")
            st.rerun()
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📤 Upload Documents", 
        "💬 Smart Q&A", 
        "🔍 Advanced Search", 
        "🧠 Document Intelligence", 
        "🔗 Cross-References",
        "📊 Evaluation Suite",
        "🚀 Search Analytics",
        "📈 Performance Dashboard",
        "🎨 Multi-Modal Elements",
        "🕸️ Knowledge Graph"
    ])
    
    with tab1:
        st.header("Upload Documents")
        st.write("Upload PDF, TXT, DOCX, or Markdown files to build your knowledge base.")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, MD"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files:")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size:,} bytes)")
            
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Save uploaded files
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = save_uploaded_file(uploaded_file)
                        file_paths.append(file_path)
                    
                    # Process documents with enhanced RAG
                    result = st.session_state.enhanced_rag.process_documents(file_paths)
                    
                    if result["success"]:
                        st.success(f"✅ Successfully processed {result['processed_files']} files!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Files Processed", result['processed_files'])
                        with col2:
                            st.metric("Total Chunks", result['total_chunks'])
                        with col3:
                            st.metric("Processing Time", f"{result['processing_time']}s")
                        
                        if result.get('invalid_files'):
                            st.warning(f"⚠️ Could not process: {', '.join([Path(f).name for f in result['invalid_files']])}")
                        
                        # Handle insights from enhanced processing
                        if result.get("insights_generated"):
                            st.session_state.document_insights = result["insights"]
                            st.success("🧠 Document intelligence analysis completed!")
                        
                        if result.get("relationships_generated"):
                            st.session_state.relationship_analysis = result["relationships"]
                            st.success("🔗 Cross-reference analysis completed!")
                        
                        # Generate smart suggestions
                        st.session_state.smart_suggestions = st.session_state.enhanced_rag.get_smart_suggestions()
                        if st.session_state.smart_suggestions:
                            st.success("💡 Smart suggestions generated!")
                        
                        # Show collection info
                        with st.expander("Vector Store Details"):
                            st.json(result['collection_info'])
                    
                    else:
                        st.error(f"❌ Processing failed: {result['error']}")
    
    with tab2:
        st.header("💡 Smart Q&A with Intelligent Assistance")
        
        # Check if documents are loaded
        status = st.session_state.enhanced_rag.get_system_status()
        if status['documents_processed'] == 0:
            st.warning("⚠️ No documents loaded. Please upload and process documents first.")
        else:
            doc_count = status['documents_processed']
            st.info(f"📄 Ready to answer questions from {doc_count} document chunks in **{st.session_state.system_mode.title()} Mode**")
            
            # Show smart suggestions
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("💡 Smart Suggestions")
                
                # Show smart suggestions from the enhanced system
                if st.session_state.smart_suggestions:
                    st.write("**🎯 Question Ideas:**")
                    for i, suggestion in enumerate(st.session_state.smart_suggestions[:4]):
                        if st.button(
                            f"❓ {suggestion['question'][:40]}{'...' if len(suggestion['question']) > 40 else ''}", 
                            key=f"suggestion_{i}",
                            help=f"Type: {suggestion['type']}, Confidence: {suggestion['confidence']:.1%}"
                        ):
                            st.session_state.suggested_question = suggestion['question']
                            st.rerun()
                
                # Refresh suggestions button
                if st.button("🔄 Refresh Suggestions", key="refresh_suggestions"):
                    st.session_state.smart_suggestions = st.session_state.enhanced_rag.get_smart_suggestions()
                    st.rerun()
                
                # Show document type info
                if st.session_state.document_insights:
                    stats = st.session_state.document_insights.get('document_statistics', {})
                    doc_types = stats.get('document_types', {})
                    if doc_types:
                        st.write("**📁 Document Types:**")
                        for doc_type, count in doc_types.items():
                            st.write(f"• {doc_type}: {count}")
            
            with col1:
                # Question input with smart assistance
                question = st.text_input(
                    "Ask a question about your documents:",
                    placeholder="e.g., What are the main topics discussed in the documents?",
                    key="question_input",
                    value=st.session_state.get('suggested_question', '')
                )
                
                # Clear suggestion after use
                if 'suggested_question' in st.session_state:
                    del st.session_state.suggested_question
            
            if st.button("🔍 Get Smart Answer", type="primary") and question:
                with st.spinner("Generating intelligent answer..."):
                    result = st.session_state.enhanced_rag.ask_question(
                        question, 
                        use_conversation=use_conversation
                    )
                    
                    if result["success"]:
                        # Display answer
                        st.subheader("💡 Answer")
                        
                        # Show enhanced answer if available
                        if result.get("enhanced_answer") and result.get("query_rewrite_applied"):
                            st.success("🎯 Enhanced answer using query intelligence:")
                            st.write(result["enhanced_answer"])
                            
                            with st.expander("📝 Original Answer (for comparison)"):
                                st.write(result["answer"])
                        else:
                            st.write(result["answer"])
                        
                        # Advanced Query Intelligence Analysis
                        if result.get("advanced_query_analysis"):
                            with st.expander("🧠 Advanced Query Intelligence Analysis"):
                                analysis = result["advanced_query_analysis"]
                                
                                # Intent Analysis
                                intent_data = analysis["intent_analysis"]
                                st.write("**🎯 Intent Analysis:**")
                                col1_int, col2_int, col3_int = st.columns(3)
                                with col1_int:
                                    st.metric("Primary Intent", intent_data["primary_intent"].title())
                                with col2_int:
                                    st.metric("Confidence", f"{intent_data['confidence']:.1%}")
                                with col3_int:
                                    st.metric("Complexity", intent_data["complexity_level"].title())
                                
                                # Query Enhancements
                                if analysis.get("query_rewrites"):
                                    st.write("**✨ Query Enhancement Suggestions:**")
                                    for i, rewrite in enumerate(analysis["query_rewrites"][:2]):
                                        st.write(f"**{i+1}. {rewrite['rewrite_type'].title()}** (Confidence: {rewrite['confidence']:.1%})")
                                        st.write(f"• Enhanced: *{rewrite['rewritten_query']}*")
                                        st.write(f"• Reasoning: {rewrite['reasoning']}")
                                        st.write("")
                                
                                # Improvement Potential
                                if analysis.get("improvement_potential"):
                                    st.write("**📈 Query Improvement Potential:**")
                                    potential = analysis["improvement_potential"]
                                    for aspect, score in potential.items():
                                        if score > 0.3:  # Only show significant improvement areas
                                            st.progress(score, text=f"{aspect.title()}: {score:.1%}")
                        
                        # Basic Query Enhancement (fallback)
                        elif result.get("query_enhancement"):
                            with st.expander("🔍 Query Enhancement Details"):
                                enhancement = result["query_enhancement"]
                                
                                # Query Analysis
                                if "query_analysis" in enhancement:
                                    analysis = enhancement["query_analysis"]
                                    st.write("**📊 Query Analysis:**")
                                    st.write(f"• Primary Intent: {analysis['primary_intent'].title()}")
                                    st.write(f"• Complexity: {analysis['complexity']}")
                                    st.write(f"• Estimated Difficulty: {analysis['estimated_difficulty']}")
                                
                                # Suggestions
                                if "intelligent_suggestions" in enhancement:
                                    suggestions = enhancement["intelligent_suggestions"][:3]
                                    if suggestions:
                                        st.write("**💡 Query Suggestions:**")
                                        for suggestion in suggestions:
                                            st.write(f"• {suggestion['suggestion']} (Type: {suggestion['type']})")
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Response Time", f"{result['response_time']}s")
                        with col2:
                            st.metric("Sources Used", result['source_count'])
                        with col3:
                            st.metric("Conversation", "Yes" if use_conversation else "No")
                        with col4:
                            rewrite_status = "✅ Applied" if result.get("query_rewrite_applied") else "❌ Not Used"
                            st.metric("Query Enhancement", rewrite_status)
                        
                        # Display sources
                        if result["sources"]:
                            display_sources(result["sources"])
                        
                        # Add to chat history
                        chat_entry = {
                            "question": question,
                            "answer": result.get("enhanced_answer", result["answer"]),
                            "sources": len(result["sources"]),
                            "response_time": result["response_time"]
                        }
                        
                        # Add intelligence metadata if available
                        if result.get("advanced_query_analysis"):
                            chat_entry["intent"] = result["advanced_query_analysis"]["intent_analysis"]["primary_intent"]
                            chat_entry["confidence"] = result["advanced_query_analysis"]["intent_analysis"]["confidence"]
                        
                        st.session_state.chat_history.append(chat_entry)
                    
                    else:
                        st.error(f"❌ Error: {result['error']}")
            
            # Display conversation history
            if st.session_state.chat_history:
                st.subheader("💬 Recent Questions")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                    with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                        st.write("**Question:**", chat['question'])
                        st.write("**Answer:**", chat['answer'])
                        st.write(f"**Sources:** {chat['sources']} | **Time:** {chat['response_time']}s")
    
    with tab3:
        st.header("🔍 Advanced Search & Retrieval")
        st.write("Pro-level search with hybrid methods, reranking, and detailed analytics.")
        
        # Check system capabilities
        status = st.session_state.enhanced_rag.get_system_status()
        capabilities = status.get('capabilities', {})
        
        if status['documents_processed'] == 0:
            st.warning("⚠️ No documents loaded. Please upload and process documents first.")
            return
        
        # Search configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Search query:",
                placeholder="Enter keywords or phrases to search for...",
                key="advanced_search_input"
            )
        
        with col2:
            search_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        # Advanced options
        st.subheader("🚀 Pro-Level Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Search method selection
            search_methods = ["hybrid", "bm25", "vector"]
            if capabilities.get('hybrid_search'):
                method_labels = {
                    "hybrid": "🎯 Hybrid (BM25 + Vector + RRF)",
                    "bm25": "📝 BM25 (Lexical Search)",
                    "vector": "🧠 Vector (Semantic Search)"
                }
                available_methods = search_methods
            else:
                method_labels = {"hybrid": "🔍 Basic Search"}
                available_methods = ["hybrid"]
            
            selected_method = st.selectbox(
                "Search Method:",
                available_methods,
                format_func=lambda x: method_labels.get(x, x),
                help="• **Hybrid**: Combines BM25 lexical + vector semantic search\n• **BM25**: Traditional keyword-based search\n• **Vector**: AI-powered semantic similarity"
            )
        
        with col2:
            # Reranking option
            enable_reranking = st.checkbox(
                "🎭 Enable Reranking",
                value=capabilities.get('reranking', False),
                disabled=not capabilities.get('reranking', False),
                help="Uses cross-encoder models to improve result ranking"
            )
        
        with col3:
            # Query expansion
            enable_expansion = st.checkbox(
                "📈 Query Expansion",  
                value=True,
                help="Automatically expands query with synonyms and variations"
            )
        
        # Search execution
        if st.button("🚀 Advanced Search", type="primary") and search_query:
            with st.spinner(f"Performing {method_labels.get(selected_method, selected_method)} search..."):
                # Execute search with advanced options
                result = st.session_state.enhanced_rag.search_documents_hybrid(
                    query=search_query,
                    k=search_k, 
                    method=selected_method,
                    enable_reranking=enable_reranking
                )
                
                if result["success"]:
                    # Display results summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📄 Results Found", result['result_count'])
                    with col2:
                        st.metric("🎯 Method Used", result['method'].upper())
                    with col3:
                        rerank_status = "✅ Applied" if result.get('reranking_applied') else "❌ Not Used"
                        st.metric("🎭 Reranking", rerank_status)
                    with col4:
                        st.metric("📊 Query", f"{len(search_query)} chars")
                    
                    st.divider()
                    
                    # Display search results
                    st.subheader("🎯 Search Results")
                    
                    for i, doc_result in enumerate(result["results"]):
                        # Prepare result title
                        filename = doc_result['metadata'].get('filename', f'Document_{i+1}')
                        score = doc_result.get('score', 0)
                        rank = doc_result.get('rank', i+1)
                        
                        # Show reranking info if available
                        rerank_info = doc_result.get('reranking_info', {})
                        title_suffix = ""
                        
                        if rerank_info:
                            original_rank = rerank_info.get('original_rank', rank)
                            if original_rank != rank:
                                title_suffix = f" (🎭 Reranked: {original_rank}→{rank})"
                        
                        result_title = f"#{rank}: {filename} (Score: {score:.3f}){title_suffix}"
                        
                        with st.expander(result_title):
                            # Content
                            st.write("**📄 Content:**")
                            st.write(doc_result["content"])
                            
                            # Advanced metrics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**📊 Search Metrics:**")
                                st.write(f"• **Source:** {doc_result.get('source', 'unknown')}")
                                st.write(f"• **Rank:** #{rank}")  
                                st.write(f"• **Score:** {score:.4f}")
                                
                                # Show search metadata
                                search_meta = doc_result.get('search_metadata', {})
                                if search_meta:
                                    st.write("**🔍 Search Details:**")
                                    for key, value in search_meta.items():
                                        if key not in ['original_metadata']:
                                            st.write(f"• **{key.title()}:** {value}")
                            
                            with col2:
                                # Reranking details
                                if rerank_info:
                                    st.write("**🎭 Reranking Analysis:**")
                                    st.write(f"• **Original Rank:** #{rerank_info.get('original_rank', 'N/A')}")
                                    st.write(f"• **Original Score:** {rerank_info.get('original_score', 0):.4f}")
                                    st.write(f"• **Confidence:** {rerank_info.get('confidence', 0):.2%}")
                                    
                                    improvement = rerank_info.get('score_improvement', 0)
                                    improvement_icon = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                                    st.write(f"• **Score Change:** {improvement_icon} {improvement:+.4f}")
                            
                            # Document metadata
                            st.write("**📋 Document Metadata:**")
                            metadata = doc_result.get('metadata', {})
                            if metadata:
                                metadata_display = {k: v for k, v in metadata.items() if k != 'chunk_content'}
                                st.json(metadata_display)
                
                else:
                    st.error(f"❌ Advanced search failed: {result.get('error', 'Unknown error')}")
        
        # Search tips
        with st.expander("💡 Pro Search Tips"):
            st.markdown("""
            **🎯 Search Method Guide:**
            - **Hybrid**: Best overall performance, combines keyword + semantic search
            - **BM25**: Great for exact matches, technical terms, and specific keywords  
            - **Vector**: Excellent for conceptual queries and semantic understanding
            
            **🎭 Reranking Benefits:**
            - Improves relevance by 15-30% on average
            - Better handling of complex queries
            - More accurate ranking of similar documents
            
            **📈 Query Optimization:**
            - Use specific terms for BM25 search
            - Use natural language for vector search
            - Hybrid method works well with both approaches
            """)
        
        # Advanced Query Intelligence Section
        st.divider()
        st.subheader("🧠 Advanced Query Intelligence")
        
        if search_query:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎯 Analyze Query Intent", help="Analyze query intent and suggest improvements"):
                    with st.spinner("Analyzing query intelligence..."):
                        try:
                            # Get advanced query analysis
                            from src.advanced_query_intelligence import AdvancedQueryIntelligence
                            query_ai = AdvancedQueryIntelligence()
                            
                            # Create context
                            context = {}
                            if st.session_state.document_insights:
                                context['document_themes'] = {'main_concepts': [c['concept'] for c in st.session_state.document_insights.get('key_concepts', [])[:5]]}
                            
                            analysis = query_ai.process_query(search_query, context)
                            
                            # Display intent analysis
                            st.subheader("🎯 Intent Analysis")
                            intent_data = analysis["intent_analysis"]
                            
                            col1_int, col2_int, col3_int = st.columns(3)
                            with col1_int:
                                st.metric("Primary Intent", intent_data["primary_intent"].title())
                            with col2_int:
                                st.metric("Confidence", f"{intent_data['confidence']:.1%}")
                            with col3_int:
                                st.metric("Complexity", intent_data["complexity_level"].title())
                            
                            # Sub-intents
                            if intent_data.get("sub_intents"):
                                st.write("**🎭 Sub-Intents Detected:**")
                                sub_intents = sorted(intent_data["sub_intents"].items(), key=lambda x: x[1], reverse=True)[:3]
                                for intent, score in sub_intents:
                                    st.progress(score, text=f"{intent.title()}: {score:.1%}")
                            
                            # Query rewrites
                            if analysis.get("query_rewrites"):
                                st.subheader("✨ Query Enhancements")
                                rewrites = analysis["query_rewrites"][:3]
                                
                                for i, rewrite in enumerate(rewrites):
                                    with st.expander(f"🔄 Enhancement #{i+1}: {rewrite['rewrite_type'].title()} (Confidence: {rewrite['confidence']:.1%})"):
                                        st.write(f"**Original:** {rewrite['original_query']}")
                                        st.write(f"**Enhanced:** {rewrite['rewritten_query']}")
                                        st.write(f"**Reasoning:** {rewrite['reasoning']}")
                                        
                                        if rewrite.get('improvement_factors'):
                                            st.write("**Improvements:**")
                                            for factor in rewrite['improvement_factors']:
                                                st.write(f"• {factor}")
                            
                            # Enhancement strategies
                            if analysis.get("enhancement_strategies"):
                                st.subheader("💡 Enhancement Strategies")
                                for strategy in analysis["enhancement_strategies"]:
                                    with st.expander(f"📈 {strategy['strategy'].title()}"):
                                        st.write(strategy['description'])
                                        if 'example' in strategy:
                                            st.code(strategy['example'])
                        
                        except Exception as e:
                            st.error(f"❌ Query intelligence analysis failed: {str(e)}")
            
            with col2:
                if st.button("📊 Get Query Analytics", help="View query processing patterns and trends"):
                    with st.spinner("Gathering query analytics..."):
                        analytics_result = st.session_state.enhanced_rag.get_query_analytics()
                        
                        if analytics_result.get("success"):
                            analytics = analytics_result["analytics"]
                            
                            if analytics.get("total_queries_processed", 0) > 0:
                                st.subheader("📈 Query Analytics Dashboard")
                                
                                # Basic metrics
                                col1_a, col2_a, col3_a = st.columns(3)
                                with col1_a:
                                    st.metric("Total Queries", analytics["total_queries_processed"])
                                with col2_a:
                                    st.metric("Avg Confidence", f"{analytics['average_confidence']:.1%}")
                                with col3_a:
                                    st.metric("Rewrite Success", f"{analytics['rewrite_success_rate']:.1%}")
                                
                                # Intent distribution
                                if analytics.get("intent_distribution"):
                                    st.write("**🎯 Intent Distribution:**")
                                    intent_dist = analytics["intent_distribution"]
                                    for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                                        st.progress(count / analytics["total_queries_processed"], text=f"{intent.title()}: {count}")
                                
                                # Complexity distribution
                                if analytics.get("complexity_distribution"):
                                    st.write("**🔧 Complexity Levels:**")
                                    complexity_dist = analytics["complexity_distribution"]
                                    for level, count in complexity_dist.items():
                                        pct = count / analytics["total_queries_processed"]
                                        st.progress(pct, text=f"{level.title()}: {count} ({pct:.1%})")
                                
                                # Recent trends
                                if analytics.get("recent_trends") and "significant_changes" in analytics["recent_trends"]:
                                    st.write("**📊 Recent Trends:**")
                                    trends = analytics["recent_trends"]["significant_changes"]
                                    for intent, trend_data in trends.items():
                                        direction = "📈" if trend_data["direction"] == "increasing" else "📉"
                                        st.write(f"{direction} {intent.title()}: {trend_data['change']:+.1%}")
                            else:
                                st.info("🔍 No query history available yet. Try running some searches first!")
                        else:
                            st.error(f"❌ Analytics failed: {analytics_result.get('error', 'Unknown error')}")

        # Quick search analytics
        st.divider()
        if st.button("📊 Quick Search Analytics") and search_query:
            with st.spinner("Analyzing search performance across methods..."):
                analytics_result = st.session_state.enhanced_rag.get_search_analytics(search_query, k=search_k)
                
                if analytics_result.get("success"):
                    analytics = analytics_result["analytics"]
                    
                    st.subheader("📊 Search Method Comparison")
                    
                    # Method comparison
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**📝 BM25 Results:**")
                        bm25_count = analytics["results_count"]["bm25"]
                        st.metric("Results", bm25_count)
                        
                        bm25_scores = analytics["score_ranges"]["bm25"]
                        if bm25_count > 0:
                            st.write(f"Score Range: {bm25_scores['min']:.3f} - {bm25_scores['max']:.3f}")
                    
                    with col2:
                        st.write("**🧠 Vector Results:**")
                        vector_count = analytics["results_count"]["vector"]
                        st.metric("Results", vector_count)
                        
                        vector_scores = analytics["score_ranges"]["vector"]
                        if vector_count > 0:
                            st.write(f"Score Range: {vector_scores['min']:.3f} - {vector_scores['max']:.3f}")
                    
                    with col3:
                        st.write("**🎯 Hybrid Results:**")
                        hybrid_count = analytics["results_count"]["hybrid"]
                        st.metric("Results", hybrid_count)
                        
                        hybrid_scores = analytics["score_ranges"]["hybrid"]
                        if hybrid_count > 0:
                            st.write(f"Score Range: {hybrid_scores['min']:.3f} - {hybrid_scores['max']:.3f}")
                    
                    # Overlap analysis
                    st.subheader("🔄 Method Overlap Analysis")
                    overlap = analytics["overlap_metrics"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("BM25 ∩ Vector", overlap["bm25_vector_overlap"])
                    with col2:
                        st.metric("BM25 Unique", overlap["bm25_unique"])
                    with col3:
                        st.metric("Vector Unique", overlap["vector_unique"])
                
                else:
                    st.error("❌ Analytics failed")

    with tab4:
        st.header("🧠 Document Intelligence & Insights")
        
        if st.session_state.document_insights:
            insights = st.session_state.document_insights
            
            # Executive Summary
            st.subheader("📋 Executive Summary")
            summary = insights.get('summary', {})
            if summary and not isinstance(summary, str):
                st.info(summary.get('executive_summary', 'No summary available'))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", summary.get('word_count', 0))
                with col2:
                    st.metric("Key Points", len(summary.get('key_points', [])))
                with col3:
                    st.metric("Confidence", f"{summary.get('confidence_score', 0):.1%}")
            
            # Key Concepts
            st.subheader("🔑 Key Concepts")
            concepts = insights.get('key_concepts', [])
            if concepts:
                concept_cols = st.columns(min(4, len(concepts)))
                for i, concept in enumerate(concepts[:4]):
                    with concept_cols[i]:
                        st.metric(
                            concept['concept'].title(), 
                            f"{concept['frequency']}x",
                            f"{concept['importance']:.1%}"
                        )
                
                # Show all concepts in expandable section
                with st.expander("View All Key Concepts"):
                    for concept in concepts:
                        concept_text = concept.get('concept', 'Unknown')
                        concept_type = concept.get('type', 'term')
                        frequency = concept.get('frequency', 0)
                        importance = concept.get('importance', 0)
                        st.write(f"**{concept_text}** ({concept_type}) - Frequency: {frequency}, Importance: {importance:.1%}")
            
            # Document Statistics
            st.subheader("📊 Document Analysis")
            stats = insights.get('document_statistics', {})
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Documents", stats.get('total_documents', 0))
                with col2:
                    st.metric("Total Words", f"{stats.get('total_words', 0):,}")
                with col3:
                    st.metric("Unique Words", f"{stats.get('unique_words', 0):,}")
                with col4:
                    st.metric("Vocabulary Richness", f"{stats.get('vocabulary_richness', 0):.1%}")
            
            # Named Entities
            entities = insights.get('named_entities', {})
            if entities and any(entities.values()):
                st.subheader("🏷️ Named Entities")
                entity_tabs = st.tabs(["👥 People", "🏢 Organizations", "📍 Locations", "📅 Dates", "💰 Money"])
                
                entity_types = ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'MONEY']
                for i, entity_type in enumerate(entity_types):
                    with entity_tabs[i]:
                        entity_list = entities.get(entity_type, [])
                        if entity_list:
                            for entity in entity_list[:10]:  # Show top 10
                                st.write(f"• {entity}")
                        else:
                            st.write("No entities found")
            
            # Complexity Analysis
            complexity = insights.get('complexity_analysis', {})
            if complexity and not isinstance(complexity, str):
                st.subheader("📈 Complexity Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Flesch Score", complexity.get('flesch_score', 0))
                    st.write(f"**Level:** {complexity.get('complexity_level', 'Unknown')}")
                    st.write(f"**Audience:** {complexity.get('recommended_audience', 'General')}")
                
                with col2:
                    st.metric("Avg Words/Sentence", complexity.get('avg_words_per_sentence', 0))
                    st.metric("Avg Syllables/Word", complexity.get('avg_syllables_per_word', 0))
            
            # Topic Clusters
            clusters = insights.get('topic_clusters', {})
            if clusters and 'clusters' in clusters:
                st.subheader("🎯 Topic Clusters")
                cluster_data = clusters['clusters']
                
                for cluster_name, cluster_info in cluster_data.items():
                    with st.expander(f"{cluster_name} ({cluster_info['document_count']} documents)"):
                        st.write("**Top Terms:**")
                        st.write(", ".join(cluster_info['top_terms'][:10]))
                        st.write("**Representative Text:**")
                        st.write(cluster_info['representative_snippet'])
            
            # Quality Assessment
            quality = insights.get('quality_assessment', {})
            if quality and not isinstance(quality, str):
                st.subheader("✅ Quality Assessment")
                
                col1, col2 = st.columns(2)
                with col1:
                    score = quality.get('overall_score', 0)
                    st.metric("Quality Score", f"{score}/100")
                    st.write(f"**Level:** {quality.get('quality_level', 'Unknown')}")
                
                with col2:
                    st.metric("Metadata Completeness", f"{quality.get('metadata_completeness', 0):.1f}%")
                    st.metric("Content Diversity", f"{quality.get('content_diversity', 0):.1f}%")
                
                if quality.get('issues_detected'):
                    st.write("**Issues Detected:**")
                    for issue in quality['issues_detected']:
                        st.warning(f"⚠️ {issue}")
                
                if quality.get('recommendations'):
                    st.write("**Recommendations:**")
                    for rec in quality['recommendations']:
                        st.info(f"💡 {rec}")
        
        else:
            st.info("📄 Upload and process documents to see intelligent insights")
            st.write("The Document Intelligence system will analyze your documents and provide:")
            st.write("• Executive summaries and key insights")
            st.write("• Named entity extraction")
            st.write("• Complexity and readability analysis")
            st.write("• Topic clustering and themes")
            st.write("• Quality assessment and recommendations")

    with tab5:
        st.header("🔗 Cross-Reference Analysis")
        
        if st.session_state.relationship_analysis:
            analysis = st.session_state.relationship_analysis
            
            # Relationship Overview
            st.subheader("🌐 Relationship Overview")
            insights = analysis.get('insights', {})
            if insights:
                st.info(insights.get('summary', 'No summary available'))
                
                # Key findings
                findings = insights.get('key_findings', [])
                if findings:
                    st.write("**Key Findings:**")
                    for finding in findings:
                        st.write(f"• {finding}")
            
            # Document Similarities
            similarities = analysis.get('similarity_matrix', {})
            if similarities and 'high_similarity_pairs' in similarities:
                st.subheader("🔍 Document Similarities")
                pairs = similarities['high_similarity_pairs']
                
                if pairs:
                    for pair in pairs:
                        with st.expander(f"{pair['doc1_name']} ↔ {pair['doc2_name']} (Similarity: {pair['similarity_score']:.3f})"):
                            st.write(f"**Relationship Strength:** {pair['relationship_strength']}")
                            st.progress(pair['similarity_score'])
                else:
                    st.info("No high similarity pairs found")
            
            # Contradictions
            contradictions = analysis.get('contradictions', [])
            if contradictions:
                st.subheader("⚠️ Potential Contradictions")
                for contradiction in contradictions:
                    with st.expander(f"⚠️ {contradiction['doc1_name']} vs {contradiction['doc2_name']} ({contradiction['contradiction_count']} conflicts)"):
                        for statement in contradiction['contradictory_statements']:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Statement 1:**")
                                st.write(statement['statement1'])
                            with col2:
                                st.write("**Statement 2:**")
                                st.write(statement['statement2'])
                            st.write(f"*Confidence: {statement.get('confidence', 0):.2f}*")
                            st.divider()
            
            # Agreements
            agreements = analysis.get('agreements', [])
            if agreements:
                st.subheader("✅ Supporting Relationships")
                for agreement in agreements:
                    with st.expander(f"✅ {agreement['doc1_name']} supports {agreement['doc2_name']} ({agreement['agreement_count']} connections)"):
                        for statement in agreement['supporting_statements']:
                            st.write("**Supporting Evidence:**")
                            st.success(statement['statement1'])
                            st.success(statement['statement2'])
                            st.write(f"*Type: {statement['agreement_type']}, Confidence: {statement.get('confidence', 0):.2f}*")
                            st.divider()
            
            # Relationship Graph
            graph = analysis.get('relationship_graph', {})
            if graph and 'edges' in graph:
                st.subheader("📊 Relationship Network")
                
                # Network statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Relationships", graph.get('total_relationships', 0))
                with col2:
                    st.metric("Network Density", f"{graph.get('network_density', 0):.2%}")
                with col3:
                    most_connected = graph.get('most_connected_documents', [])
                    if most_connected:
                        st.metric("Most Connected", f"Doc {most_connected[0][0]}")
                
                # Show relationship types
                edges = graph.get('edges', [])
                if edges:
                    relationship_types = {}
                    for edge in edges:
                        rel_type = edge['type']
                        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                    
                    st.write("**Relationship Types:**")
                    for rel_type, count in relationship_types.items():
                        st.write(f"• {rel_type.title()}: {count}")
            
            # Temporal Analysis
            temporal = analysis.get('temporal_analysis', {})
            if temporal and temporal.get('document_temporal_info'):
                st.subheader("⏰ Temporal Analysis")
                
                temporal_info = temporal['document_temporal_info']
                if temporal_info:
                    for doc_info in temporal_info:
                        st.write(f"**{doc_info['document_name']}:**")
                        st.write(f"Years mentioned: {', '.join(map(str, doc_info['years_mentioned']))}")
                        st.write(f"Time span: {doc_info['temporal_span']} years")
                
                # Show temporal overlaps
                overlaps = temporal.get('temporal_overlaps', [])
                if overlaps:
                    st.write("**Temporal Overlaps:**")
                    for overlap in overlaps:
                        st.write(f"• {overlap['doc1_name']} and {overlap['doc2_name']}: {', '.join(map(str, overlap['common_years']))}")
        
        else:
            st.info("📄 Upload multiple documents to see cross-reference analysis")
            st.write("The Cross-Reference system will analyze:")
            st.write("• Document similarities and relationships")
            st.write("• Contradictions and conflicts")
            st.write("• Supporting evidence and agreements")
            st.write("• Citation patterns and references")
            st.write("• Temporal relationships and overlaps")

    with tab6:
        st.header("📊 RAG Evaluation Suite")
        st.write("Comprehensive evaluation framework for measuring RAG system performance.")
        
        # Check system capabilities
        status = st.session_state.enhanced_rag.get_system_status()
        capabilities = status.get('capabilities', {})
        
        if not capabilities.get('comprehensive_evaluation'):
            st.error("❌ Evaluation framework not available")
            return
        
        if status['documents_processed'] == 0:
            st.warning("⚠️ No documents loaded. Please upload and process documents first.")
            return
        
        # Evaluation options
        eval_tabs = st.tabs(["🔍 Single Query Evaluation", "📈 Batch Evaluation", "📊 Evaluation History"])
        
        with eval_tabs[0]:
            st.subheader("🔍 Single Query Evaluation")
            st.write("Evaluate individual queries with comprehensive metrics.")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                eval_query = st.text_input(
                    "Query to evaluate:",
                    placeholder="Enter a question to evaluate system performance...",
                    key="eval_query"
                )
            
            with col2:
                eval_method = st.selectbox(
                    "Evaluation Method:",
                    ["hybrid", "bm25", "vector"],
                    format_func=lambda x: {
                        "hybrid": "🎯 Hybrid Search",
                        "bm25": "📝 BM25 Search", 
                        "vector": "🧠 Vector Search"
                    }.get(x, x)
                )
            
            # Optional ground truth
            ground_truth = st.text_area(
                "Ground Truth Answer (Optional):",
                placeholder="Enter the expected correct answer for comparison...",
                height=100
            )
            
            if st.button("🧪 Evaluate Query", type="primary") and eval_query:
                with st.spinner("Running comprehensive evaluation..."):
                    eval_result = st.session_state.enhanced_rag.evaluate_query(
                        query=eval_query,
                        ground_truth_answer=ground_truth if ground_truth.strip() else None,
                        method=eval_method,
                        enable_reranking=True
                    )
                    
                    if eval_result.get("success"):
                        result = eval_result["evaluation_result"]
                        
                        # Overall metrics
                        st.subheader("📊 Overall Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Score", f"{result.overall_score:.2%}")
                        with col2:
                            st.metric("Evaluation Time", f"{eval_result['evaluation_time']:.2f}s")
                        with col3:
                            st.metric("Method Used", eval_method.upper())
                        with col4:
                            st.metric("Sources Retrieved", len(result.retrieved_documents))
                        
                        # Detailed metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🔍 Retrieval Metrics")
                            
                            # Precision@K metrics
                            precision_data = result.retrieval_metrics.precision_at_k
                            for k, precision in precision_data.items():
                                st.metric(f"Precision@{k}", f"{precision:.2%}")
                            
                            # Other retrieval metrics
                            st.metric("Mean Reciprocal Rank", f"{result.retrieval_metrics.mean_reciprocal_rank:.3f}")
                            st.metric("Coverage", f"{result.retrieval_metrics.coverage:.2%}")
                            st.metric("Retrieval Latency", f"{result.retrieval_metrics.retrieval_latency:.3f}s")
                        
                        with col2:
                            st.subheader("💬 Generation Metrics")
                            
                            # Generation quality metrics
                            gen_metrics = result.generation_metrics
                            st.metric("Faithfulness", f"{gen_metrics.faithfulness:.2%}")
                            st.metric("Answer Relevancy", f"{gen_metrics.answer_relevancy:.2%}")
                            st.metric("Answer Correctness", f"{gen_metrics.answer_correctness:.2%}")
                            st.metric("Coherence Score", f"{gen_metrics.coherence_score:.2%}")
                            st.metric("Hallucination Risk", f"{gen_metrics.hallucination_score:.2%}", delta=f"{-gen_metrics.hallucination_score:.2%}")
                            st.metric("Generation Latency", f"{gen_metrics.generation_latency:.3f}s")
                        
                        # Quality metrics
                        st.subheader("✨ Text Quality Metrics")
                        col1, col2, col3 = st.columns(3)
                        
                        quality_metrics = result.quality_metrics
                        
                        with col1:
                            if ground_truth:
                                st.metric("BLEU Score", f"{quality_metrics.bleu_score:.3f}")
                                rouge_scores = quality_metrics.rouge_scores
                                st.metric("ROUGE-1", f"{rouge_scores.get('rouge-1', 0):.3f}")
                                st.metric("ROUGE-2", f"{rouge_scores.get('rouge-2', 0):.3f}")
                        
                        with col2:
                            st.metric("Fluency Score", f"{quality_metrics.fluency_score:.2%}")
                            st.metric("Readability Score", f"{quality_metrics.readability_score:.2%}")
                            st.metric("Diversity Score", f"{quality_metrics.diversity_score:.2%}")
                        
                        with col3:
                            st.metric("Factual Consistency", f"{quality_metrics.factual_consistency:.2%}")
                            st.metric("Citation Accuracy", f"{quality_metrics.citation_accuracy:.2%}")
                        
                        # Generated answer
                        st.subheader("💡 Generated Answer")
                        st.info(eval_result["generated_answer"])
                        
                        # Performance insights
                        with st.expander("🔬 Detailed Analysis"):
                            st.write("**Evaluation Metadata:**")
                            st.json({
                                "query": result.query,
                                "timestamp": result.timestamp,
                                "evaluation_time": result.evaluation_time,
                                "metadata": result.metadata
                            })
                    
                    else:
                        st.error(f"❌ Evaluation failed: {eval_result.get('error', 'Unknown error')}")
        
        with eval_tabs[1]:
            st.subheader("📈 Batch Evaluation")
            st.write("Evaluate multiple queries for comprehensive performance analysis.")
            
            # Sample evaluation dataset
            st.write("**📋 Create Evaluation Dataset:**")
            
            # Initialize dataset in session state
            if 'eval_dataset' not in st.session_state:
                st.session_state.eval_dataset = []
            
            # Add new evaluation example
            with st.expander("➕ Add Evaluation Example"):
                new_query = st.text_input("Query:", key="new_eval_query")
                new_ground_truth = st.text_area("Ground Truth Answer:", key="new_eval_ground_truth")
                new_relevant_docs = st.text_input("Relevant Document IDs (comma-separated):", key="new_eval_docs")
                
                if st.button("➕ Add to Dataset") and new_query:
                    relevant_ids = [doc.strip() for doc in new_relevant_docs.split(",") if doc.strip()] if new_relevant_docs else []
                    
                    st.session_state.eval_dataset.append({
                        "query": new_query,
                        "ground_truth_answer": new_ground_truth,
                        "relevant_document_ids": relevant_ids
                    })
                    
                    st.success(f"✅ Added query to dataset! Total: {len(st.session_state.eval_dataset)}")
                    st.rerun()
            
            # Display current dataset
            if st.session_state.eval_dataset:
                st.write(f"**📊 Current Dataset ({len(st.session_state.eval_dataset)} queries):**")
                
                for i, example in enumerate(st.session_state.eval_dataset):
                    with st.expander(f"Query {i+1}: {example['query'][:50]}..."):
                        st.write(f"**Query:** {example['query']}")
                        st.write(f"**Ground Truth:** {example['ground_truth_answer'][:100]}...")
                        st.write(f"**Relevant Docs:** {len(example['relevant_document_ids'])}")
                        
                        if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                            st.session_state.eval_dataset.pop(i)
                            st.rerun()
                
                # Run batch evaluation
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_method = st.selectbox(
                        "Evaluation Method:",
                        ["hybrid", "bm25", "vector"],
                        key="batch_method"
                    )
                
                with col2:
                    enable_batch_reranking = st.checkbox("Enable Reranking", value=True, key="batch_reranking")
                
                if st.button("🚀 Run Batch Evaluation", type="primary"):
                    with st.spinner(f"Evaluating {len(st.session_state.eval_dataset)} queries..."):
                        batch_result = st.session_state.enhanced_rag.run_evaluation_suite(
                            evaluation_dataset=st.session_state.eval_dataset,
                            method=batch_method,
                            enable_reranking=enable_batch_reranking
                        )
                        
                        if batch_result.get("success"):
                            results = batch_result["evaluation_results"]
                            
                            st.success(f"✅ Evaluated {results['successful_evaluations']}/{results['dataset_size']} queries!")
                            
                            # Aggregate metrics
                            st.subheader("📊 Aggregate Results")
                            
                            aggregate = results["aggregate_metrics"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Overall Score", f"{aggregate['overall_score']:.2%}")
                            with col2:
                                st.metric("Avg Precision@3", f"{aggregate['retrieval_metrics']['precision_at_k'].get(3, 0):.2%}")
                            with col3:
                                st.metric("Avg Faithfulness", f"{aggregate['generation_metrics']['faithfulness']:.2%}")
                            with col4:
                                st.metric("Avg Response Time", f"{results['average_time_per_query']:.2f}s")
                            
                            # Detailed aggregate metrics
                            with st.expander("📈 Detailed Aggregate Metrics"):
                                st.json(aggregate)
                        
                        else:
                            st.error(f"❌ Batch evaluation failed: {batch_result.get('error')}")
            
            else:
                st.info("📝 Add some evaluation examples to start batch evaluation")
                
                # Sample dataset button
                if st.button("📋 Load Sample Dataset"):
                    from src.evaluation import create_sample_evaluation_dataset
                    st.session_state.eval_dataset = create_sample_evaluation_dataset()
                    st.success("✅ Loaded sample evaluation dataset!")
                    st.rerun()
        
        with eval_tabs[2]:
            st.subheader("📊 Evaluation History")
            
            # Get evaluation summary
            summary_result = st.session_state.enhanced_rag.get_evaluation_summary()
            
            if summary_result.get("success"):
                summary = summary_result["summary"]
                
                if summary.get("summary"):
                    # Performance overview
                    st.write("**📈 Performance Overview:**")
                    perf_summary = summary["summary"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Evaluations", perf_summary["total_evaluations"])
                    with col2:
                        st.metric("Average Score", f"{perf_summary['average_overall_score']:.2%}")
                    with col3:
                        recent_trend = perf_summary["recent_performance_trend"]["improvement"]
                        trend_icon = "📈" if recent_trend > 0 else "📉" if recent_trend < 0 else "➡️"
                        st.metric("Trend", f"{trend_icon} {recent_trend:+.2%}")
                    
                    # Top and bottom performing queries
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🏆 Top Performing Queries:**")
                        for query in summary["top_performing_queries"]:
                            st.write(f"• {query['query'][:40]}... ({query['score']:.2%})")
                    
                    with col2:
                        st.write("**📉 Lowest Performing Queries:**")
                        for query in summary["lowest_performing_queries"]:
                            st.write(f"• {query['query'][:40]}... ({query['score']:.2%})")
                    
                    # Export results
                    if st.button("💾 Export Evaluation Results"):
                        export_result = st.session_state.enhanced_rag.export_evaluation_results("evaluation_results.json")
                        if export_result.get("success"):
                            st.success("✅ Results exported to evaluation_results.json")
                        else:
                            st.error(f"❌ Export failed: {export_result.get('error')}")
                
                else:
                    st.info("📊 No evaluation history available. Run some evaluations first!")
            
            else:
                st.error("❌ Failed to load evaluation history")

    with tab7:
        st.header("🚀 Search Analytics & Comparison")
        st.write("Deep dive into search performance across different methods and configurations.")
        
        # Check system capabilities
        status = st.session_state.enhanced_rag.get_system_status()
        capabilities = status.get('capabilities', {})
        
        if status['documents_processed'] == 0:
            st.warning("⚠️ No documents loaded. Please upload and process documents first.")
            return
        
        # Analytics tabs
        analytics_tabs = st.tabs(["📊 Method Comparison", "🎭 Reranking Analysis", "⚡ Performance Trends"])
        
        with analytics_tabs[0]:
            st.subheader("📊 Search Method Comparison")
            
            # Test query for comparison
            test_query = st.text_input(
                "Test Query for Comparison:",
                placeholder="Enter a query to compare across different search methods...",
                key="analytics_query"
            )
            
            comparison_k = st.slider("Results to Compare", min_value=1, max_value=15, value=5, key="analytics_k")
            
            if st.button("🔍 Compare Methods", type="primary") and test_query:
                with st.spinner("Running comprehensive search method comparison..."):
                    analytics_result = st.session_state.enhanced_rag.get_search_analytics(test_query, k=comparison_k)
                    
                    if analytics_result.get("success"):
                        analytics = analytics_result["analytics"]
                        
                        # Method performance overview
                        st.subheader("🎯 Method Performance Overview")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # BM25 Analysis
                        with col1:
                            st.write("**📝 BM25 (Lexical)**")
                            bm25_data = analytics["results_count"]["bm25"]
                            bm25_scores = analytics["score_ranges"]["bm25"]
                            
                            st.metric("Results Found", bm25_data)
                            if bm25_data > 0:
                                st.metric("Max Score", f"{bm25_scores['max']:.3f}")
                                st.metric("Avg Score", f"{bm25_scores['avg']:.3f}")
                                st.progress(min(1.0, bm25_scores['avg'] / 10))  # Normalize for display
                        
                        # Vector Analysis  
                        with col2:
                            st.write("**🧠 Vector (Semantic)**")
                            vector_data = analytics["results_count"]["vector"]
                            vector_scores = analytics["score_ranges"]["vector"]
                            
                            st.metric("Results Found", vector_data)
                            if vector_data > 0:
                                st.metric("Max Score", f"{vector_scores['max']:.3f}")
                                st.metric("Avg Score", f"{vector_scores['avg']:.3f}")
                                st.progress(min(1.0, vector_scores['avg']))  # Already normalized 0-1
                        
                        # Hybrid Analysis
                        with col3:
                            st.write("**🎯 Hybrid (Combined)**")
                            hybrid_data = analytics["results_count"]["hybrid"]
                            hybrid_scores = analytics["score_ranges"]["hybrid"]
                            
                            st.metric("Results Found", hybrid_data)
                            if hybrid_data > 0:
                                st.metric("Max Score", f"{hybrid_scores['max']:.3f}")
                                st.metric("Avg Score", f"{hybrid_scores['avg']:.3f}")
                                st.progress(min(1.0, hybrid_scores['avg']))
                        
                        st.divider()
                        
                        # Overlap Analysis
                        st.subheader("🔄 Result Overlap Analysis")
                        overlap = analytics["overlap_metrics"]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("BM25 ∩ Vector", overlap["bm25_vector_overlap"], 
                                     help="Documents found by both BM25 and Vector search")
                        
                        with col2:
                            st.metric("All Methods ∩", overlap["all_methods_overlap"],
                                     help="Documents found by all three methods")
                        
                        with col3:
                            st.metric("BM25 Unique", overlap["bm25_unique"],
                                     help="Documents only found by BM25")
                        
                        with col4:
                            st.metric("Vector Unique", overlap["vector_unique"],
                                     help="Documents only found by Vector search")
                        
                        # Insights
                        st.subheader("💡 Method Selection Insights")
                        
                        # Generate recommendations based on overlap
                        total_bm25 = analytics["results_count"]["bm25"]
                        total_vector = analytics["results_count"]["vector"]
                        
                        if total_bm25 > total_vector:
                            st.info("📝 **BM25 Dominance**: This query benefits from lexical/keyword matching. Consider using BM25 for similar technical or specific term queries.")
                        elif total_vector > total_bm25:
                            st.info("🧠 **Vector Advantage**: This query benefits from semantic understanding. Vector search excels with conceptual or natural language queries.")
                        else:
                            st.info("🎯 **Balanced Query**: Both methods perform similarly. Hybrid search provides the best combined approach.")
                        
                        # Detailed comparison table
                        with st.expander("📋 Detailed Method Comparison"):
                            import pandas as pd
                            
                            comparison_data = {
                                "Method": ["BM25", "Vector", "Hybrid"],
                                "Results Found": [
                                    analytics["results_count"]["bm25"],
                                    analytics["results_count"]["vector"], 
                                    analytics["results_count"]["hybrid"]
                                ],
                                "Max Score": [
                                    analytics["score_ranges"]["bm25"]["max"],
                                    analytics["score_ranges"]["vector"]["max"],
                                    analytics["score_ranges"]["hybrid"]["max"]
                                ],
                                "Avg Score": [
                                    analytics["score_ranges"]["bm25"]["avg"],
                                    analytics["score_ranges"]["vector"]["avg"],
                                    analytics["score_ranges"]["hybrid"]["avg"]
                                ],
                                "Min Score": [
                                    analytics["score_ranges"]["bm25"]["min"],
                                    analytics["score_ranges"]["vector"]["min"],
                                    analytics["score_ranges"]["hybrid"]["min"]
                                ]
                            }
                            
                            df = pd.DataFrame(comparison_data)
                            st.dataframe(df, use_container_width=True)
                    
                    else:
                        st.error("❌ Analytics failed")
        
        with analytics_tabs[1]:
            st.subheader("🎭 Reranking Analysis")
            
            if not capabilities.get('reranking'):
                st.warning("⚠️ Reranking not available. This requires cross-encoder models.")
                return
            
            st.write("Analyze the impact of reranking on search results.")
            
            rerank_query = st.text_input(
                "Query for Reranking Analysis:",
                placeholder="Enter a query to analyze reranking impact...",
                key="rerank_analysis_query"
            )
            
            if st.button("🎭 Analyze Reranking Impact", type="primary") and rerank_query:
                with st.spinner("Analyzing reranking performance..."):
                    # Get results with and without reranking
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🔍 Without Reranking**")
                        result_no_rerank = st.session_state.enhanced_rag.search_documents_hybrid(
                            query=rerank_query,
                            k=10,
                            method='hybrid',
                            enable_reranking=False
                        )
                        
                        if result_no_rerank.get("success"):
                            results_orig = result_no_rerank["results"]
                            st.metric("Results", len(results_orig))
                            
                            # Show top 3 results
                            for i, result in enumerate(results_orig[:3]):
                                st.write(f"**#{i+1}:** {result['metadata'].get('filename', 'Unknown')} (Score: {result['score']:.3f})")
                    
                    with col2:
                        st.write("**🎭 With Reranking**")
                        result_rerank = st.session_state.enhanced_rag.search_documents_hybrid(
                            query=rerank_query,
                            k=10,
                            method='hybrid',
                            enable_reranking=True
                        )
                        
                        if result_rerank.get("success"):
                            results_reranked = result_rerank["results"]
                            st.metric("Results", len(results_reranked))
                            
                            # Show top 3 results
                            for i, result in enumerate(results_reranked[:3]):
                                rerank_info = result.get('reranking_info', {})
                                orig_rank = rerank_info.get('original_rank', i+1)
                                confidence = rerank_info.get('confidence', 0)
                                
                                rank_change = f" (was #{orig_rank})" if orig_rank != i+1 else ""
                                st.write(f"**#{i+1}:** {result['metadata'].get('filename', 'Unknown')} (Score: {result['score']:.3f}){rank_change}")
                                if confidence > 0:
                                    st.write(f"    Confidence: {confidence:.1%}")
                    
                    # Reranking impact analysis
                    if (result_no_rerank.get("success") and result_rerank.get("success") and 
                        result_rerank.get("reranking_applied")):
                        
                        st.divider()
                        st.subheader("📊 Reranking Impact Analysis")
                        
                        # Calculate rank changes
                        rank_changes = []
                        score_improvements = []
                        
                        for result in results_reranked:
                            rerank_info = result.get('reranking_info', {})
                            if rerank_info:
                                orig_rank = rerank_info.get('original_rank', 0)
                                new_rank = result.get('rank', 0)
                                
                                if orig_rank > 0 and new_rank > 0:
                                    rank_change = orig_rank - new_rank  # Positive = improvement
                                    rank_changes.append(rank_change)
                                    
                                    score_improvement = rerank_info.get('score_improvement', 0)
                                    score_improvements.append(score_improvement)
                        
                        if rank_changes:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                avg_rank_change = sum(rank_changes) / len(rank_changes)
                                st.metric("Avg Rank Change", f"{avg_rank_change:+.1f}")
                            
                            with col2:
                                improved_results = sum(1 for x in rank_changes if x > 0)
                                st.metric("Results Improved", f"{improved_results}/{len(rank_changes)}")
                            
                            with col3:
                                max_improvement = max(rank_changes) if rank_changes else 0
                                st.metric("Max Improvement", f"+{max_improvement}" if max_improvement > 0 else "0")
                            
                            with col4:
                                avg_confidence = sum(r.get('reranking_info', {}).get('confidence', 0) 
                                                   for r in results_reranked) / len(results_reranked)
                                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                            
                            # Improvement breakdown
                            if score_improvements:
                                st.write("**📈 Score Improvement Distribution:**")
                                
                                positive_improvements = [x for x in score_improvements if x > 0]
                                negative_improvements = [x for x in score_improvements if x < 0]
                                neutral_improvements = [x for x in score_improvements if x == 0]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Improved", len(positive_improvements), 
                                             delta=f"+{sum(positive_improvements):.3f}" if positive_improvements else "0")
                                with col2:
                                    st.metric("Declined", len(negative_improvements),
                                             delta=f"{sum(negative_improvements):.3f}" if negative_improvements else "0")
                                with col3:
                                    st.metric("Unchanged", len(neutral_improvements))
        
        with analytics_tabs[2]:
            st.subheader("⚡ Performance Trends")
            
            # System performance metrics
            if capabilities.get('hybrid_search'):
                hybrid_status = status.get('hybrid_search_status', {})
                
                if hybrid_status.get('reranker_status', {}).get('available'):
                    reranker_stats = hybrid_status['reranker_status']['performance']
                    
                    st.write("**🎭 Reranking Performance:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Queries", reranker_stats.get('total_queries', 0))
                    with col2:
                        st.metric("Documents Reranked", reranker_stats.get('total_documents_reranked', 0))
                    with col3:
                        st.metric("Avg Latency", f"{reranker_stats.get('average_latency', 0):.3f}s")
                    with col4:
                        cache_hits = reranker_stats.get('cache_hits', 0)
                        cache_total = cache_hits + reranker_stats.get('cache_misses', 0)
                        cache_rate = (cache_hits / cache_total * 100) if cache_total > 0 else 0
                        st.metric("Cache Hit Rate", f"{cache_rate:.1f}%")
            
            # Evaluation trends if available
            eval_summary_result = st.session_state.enhanced_rag.get_evaluation_summary()
            
            if eval_summary_result.get("success"):
                eval_summary = eval_summary_result["summary"]
                
                if eval_summary.get("summary"):
                    st.divider()
                    st.write("**📊 Evaluation Performance Trends:**")
                    
                    perf_summary = eval_summary["summary"]
                    recent_trend = perf_summary["recent_performance_trend"]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Evaluations", perf_summary["total_evaluations"])
                    
                    with col2:
                        st.metric("Overall Average", f"{perf_summary['average_overall_score']:.2%}")
                    
                    with col3:
                        improvement = recent_trend["improvement"]
                        trend_label = "📈 Improving" if improvement > 0 else "📉 Declining" if improvement < 0 else "➡️ Stable"
                        st.metric("Recent Trend", trend_label, delta=f"{improvement:+.2%}")
            
            # System health indicators
            st.divider()
            st.write("**🏥 System Health Metrics:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                doc_count = status.get('documents_processed', 0)
                health_color = "🟢" if doc_count > 0 else "🔴"
                st.metric(f"{health_color} Documents", doc_count)
            
            with col2:
                chat_count = len(st.session_state.chat_history)
                activity_color = "🟢" if chat_count > 0 else "🟡"
                st.metric(f"{activity_color} Queries", chat_count)
            
            with col3:
                hybrid_ready = capabilities.get('hybrid_search', False)
                hybrid_color = "🟢" if hybrid_ready else "🟡"
                st.metric(f"{hybrid_color} Hybrid Search", "Ready" if hybrid_ready else "Basic")
            
            with col4:
                rerank_ready = capabilities.get('reranking', False)
                rerank_color = "🟢" if rerank_ready else "🟡"
                st.metric(f"{rerank_color} Reranking", "Active" if rerank_ready else "Disabled")

    with tab8:
        st.header("📈 Pro-Level Performance Dashboard")
        st.write("Real-time system performance monitoring and advanced analytics.")
        
        # System metrics
        status = st.session_state.enhanced_rag.get_system_status()
        capabilities = status.get('capabilities', {})
        
        # Performance overview
        st.subheader("⚡ System Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            doc_count = status.get('documents_processed', 0)
            st.metric("📄 Documents Processed", doc_count)
        
        with col2:
            chat_length = len(st.session_state.chat_history)
            st.metric("💬 Questions Asked", chat_length)
        
        with col3:
            suggestions_count = len(st.session_state.get('smart_suggestions', []))
            st.metric("💡 Smart Suggestions", suggestions_count)
        
        with col4:
            mode = status.get('mode', 'unknown')
            mode_display = "🌐 Online" if mode == "online" else "🔧 Offline" if mode == "offline" else "⚙️ Basic"
            st.metric("🤖 System Mode", mode_display)
        
        st.divider()
        
        # Advanced capabilities status
        st.subheader("🚀 Pro-Level Capabilities Status")
        
        capability_cols = st.columns(3)
        
        with capability_cols[0]:
            # Hybrid Search Status
            hybrid_status = "✅ Active" if capabilities.get('hybrid_search') else "❌ Disabled"
            st.metric("🎯 Hybrid Search", hybrid_status)
            
            # Show hybrid search details
            if capabilities.get('hybrid_search'):
                hybrid_info = status.get('hybrid_search_status', {})
                if hybrid_info.get('initialized'):
                    st.write(f"• Documents: {hybrid_info.get('document_count', 0)}")
                    st.write(f"• Vocabulary: {hybrid_info.get('vocabulary_size', 0):,} terms")
                    
                    weights = hybrid_info.get('weights', {})
                    st.write(f"• BM25 Weight: {weights.get('bm25', 0):.1%}")
                    st.write(f"• Vector Weight: {weights.get('vector', 0):.1%}")
        
        with capability_cols[1]:
            # Reranking Status
            rerank_status = "✅ Active" if capabilities.get('reranking') else "❌ Disabled"
            st.metric("🎭 Cross-Encoder Reranking", rerank_status)
            
            # Show reranking performance
            if capabilities.get('reranking'):
                hybrid_info = status.get('hybrid_search_status', {})
                reranker_info = hybrid_info.get('reranker_status', {})
                
                if reranker_info.get('available'):
                    performance = reranker_info.get('performance', {})
                    st.write(f"• Total Queries: {performance.get('total_queries', 0)}")
                    st.write(f"• Avg Latency: {performance.get('average_latency', 0):.3f}s")
                    
                    cache_hits = performance.get('cache_hits', 0)
                    cache_total = cache_hits + performance.get('cache_misses', 0)
                    if cache_total > 0:
                        cache_rate = cache_hits / cache_total
                        st.write(f"• Cache Hit Rate: {cache_rate:.1%}")
        
        with capability_cols[2]:
            # Evaluation Status
            eval_status = "✅ Active" if capabilities.get('comprehensive_evaluation') else "❌ Disabled"
            st.metric("📊 Evaluation Framework", eval_status)
            
            # Show evaluation stats
            if capabilities.get('comprehensive_evaluation'):
                eval_info = status.get('evaluation_status', {})
                st.write(f"• Total Evaluations: {eval_info.get('total_evaluations', 0)}")
                st.write(f"• Recent Evaluations: {eval_info.get('recent_evaluations', 0)}")
        
        # Adaptive Chunking Dashboard
        if capabilities.get('adaptive_chunking'):
            st.divider()
            st.subheader("🧩 Adaptive Chunking Analytics")
            
            chunking_info = status.get('adaptive_chunking_status', {})
            
            # Basic metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Adaptive Chunks", chunking_info.get('total_adaptive_chunks', 0))
            with col2:
                st.metric("🎭 Chunk Types", chunking_info.get('chunk_types', 0))
            with col3:
                st.metric("🏗️ Structure Aware", chunking_info.get('structure_aware', 0))
            with col4:
                avg_confidence = chunking_info.get('avg_chunk_confidence', 0)
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
            
            # Detailed analytics
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Get Chunking Analytics", help="Analyze adaptive chunking performance"):
                    with st.spinner("Analyzing adaptive chunking..."):
                        analytics_result = st.session_state.enhanced_rag.get_adaptive_chunking_analytics()
                        
                        if analytics_result.get("success"):
                            analytics = analytics_result["analytics"]
                            
                            st.subheader("📈 Chunking Performance Metrics")
                            
                            # Chunk size statistics
                            size_stats = analytics["chunk_size_stats"]
                            st.write("**📏 Chunk Size Distribution:**")
                            col1_size, col2_size, col3_size = st.columns(3)
                            with col1_size:
                                st.metric("Mean Size", f"{size_stats['mean']:.0f} chars")
                            with col2_size:
                                st.metric("Median Size", f"{size_stats['median']:.0f} chars")
                            with col3_size:
                                st.metric("Size Range", f"{size_stats['min']:.0f} - {size_stats['max']:.0f}")
                            
                            # Coherence and confidence
                            coherence_stats = analytics["coherence_stats"]
                            confidence_stats = analytics["confidence_stats"]
                            
                            st.write("**🎯 Quality Metrics:**")
                            col1_qual, col2_qual = st.columns(2)
                            with col1_qual:
                                st.metric("Avg Coherence", f"{coherence_stats['mean']:.1%}")
                                st.progress(coherence_stats['mean'], text=f"Semantic coherence: {coherence_stats['mean']:.1%}")
                            with col2_qual:
                                st.metric("Avg Confidence", f"{confidence_stats['mean']:.1%}")
                                st.progress(confidence_stats['mean'], text=f"Chunking confidence: {confidence_stats['mean']:.1%}")
                            
                            # Type and complexity distribution
                            st.write("**🎭 Chunk Type Distribution:**")
                            type_dist = analytics["type_distribution"]
                            for chunk_type, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
                                pct = count / analytics["total_chunks"]
                                st.progress(pct, text=f"{chunk_type.title()}: {count} chunks ({pct:.1%})")
                            
                            st.write("**🔧 Complexity Distribution:**")
                            complexity_dist = analytics["complexity_distribution"]
                            for complexity, count in complexity_dist.items():
                                pct = count / analytics["total_chunks"]
                                st.progress(pct, text=f"{complexity.title()}: {count} chunks ({pct:.1%})")
                            
                            # Structure depth info
                            structure_info = analytics["structure_depth"]
                            st.write("**🏗️ Document Structure:**")
                            st.write(f"• Maximum Depth: Level {structure_info['max_level']}")
                            st.write(f"• Average Level: {structure_info['avg_level']:.1f}")
                            
                            # Relationships
                            relationships = analytics["relationships"]
                            st.write("**🔗 Chunk Relationships:**")
                            st.write(f"• Total Relationships: {relationships['total_relationships']}")
                            st.write(f"• Avg per Chunk: {relationships['avg_relationships_per_chunk']:.1f}")
                            
                        else:
                            st.error(f"❌ Analytics failed: {analytics_result.get('error', 'Unknown error')}")
            
            with col2:
                if st.button("🔍 View Chunk Details", help="Browse individual chunk information"):
                    with st.spinner("Loading chunk details..."):
                        chunk_details = st.session_state.enhanced_rag.get_chunk_details()
                        
                        if chunk_details.get("success"):
                            st.subheader("📄 Chunk Browser")
                            
                            total_chunks = chunk_details["total_chunks"]
                            displayed = chunk_details["displayed_chunks"]
                            st.info(f"Showing {displayed} of {total_chunks} chunks")
                            
                            # Display chunks
                            for i, chunk in enumerate(chunk_details["chunks"]):
                                with st.expander(f"Chunk #{i+1}: {chunk['type'].title()} (Size: {chunk['size']}, Confidence: {chunk['confidence']:.1%})"):
                                    st.write(f"**ID:** {chunk['id']}")
                                    st.write(f"**Type:** {chunk['type']}")
                                    st.write(f"**Size:** {chunk['size']} characters")
                                    st.write(f"**Complexity:** {chunk['complexity']}")
                                    
                                    # Quality metrics
                                    col1_chunk, col2_chunk = st.columns(2)
                                    with col1_chunk:
                                        st.progress(chunk['coherence'], text=f"Coherence: {chunk['coherence']:.1%}")
                                    with col2_chunk:
                                        st.progress(chunk['confidence'], text=f"Confidence: {chunk['confidence']:.1%}")
                                    
                                    st.write("**Preview:**")
                                    st.write(chunk['preview'])
                        
                        else:
                            st.error(f"❌ Failed to load chunk details: {chunk_details.get('error', 'Unknown error')}")
        
        # Response time analytics
        if st.session_state.chat_history:
            st.divider()
            st.subheader("⏱️ Response Time Analytics")
            
            # Calculate response time statistics
            response_times = [chat.get('response_time', 0) for chat in st.session_state.chat_history if chat.get('response_time')]
            
            if response_times:
                import numpy as np
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_time = np.mean(response_times)
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
                
                with col2:
                    min_time = min(response_times)
                    st.metric("Fastest Response", f"{min_time:.2f}s")
                
                with col3:
                    max_time = max(response_times)
                    st.metric("Slowest Response", f"{max_time:.2f}s")
                
                with col4:
                    # Performance target (200ms from requirements)
                    under_target = sum(1 for t in response_times if t < 0.2)
                    target_rate = under_target / len(response_times)
                    st.metric("< 200ms Target", f"{target_rate:.1%}")
                
                # Response time trend visualization
                if len(response_times) > 2:
                    st.write("**📈 Response Time Trend:**")
                    
                    try:
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(range(len(response_times)), response_times, marker='o', linewidth=2, markersize=4)
                        ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='200ms Target')
                        ax.set_title('Response Time Trend', fontsize=14)
                        ax.set_xlabel('Query Number')
                        ax.set_ylabel('Response Time (seconds)')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        # Add trend line
                        z = np.polyfit(range(len(response_times)), response_times, 1)
                        p = np.poly1d(z)
                        ax.plot(range(len(response_times)), p(range(len(response_times))), "r--", alpha=0.8, label='Trend')
                        
                        st.pyplot(fig)
                        
                        # Performance insight
                        trend_slope = z[0]
                        if trend_slope < -0.01:
                            st.success("📈 Performance is improving over time!")
                        elif trend_slope > 0.01:
                            st.warning("📉 Performance is declining - consider optimization.")
                        else:
                            st.info("➡️ Performance is stable.")
                            
                    except Exception as e:
                        st.write("Chart visualization unavailable")
        
        # Cache Performance Dashboard
        if capabilities.get('high_performance_caching'):
            st.divider()
            st.subheader("🚀 High-Performance Cache Dashboard")
            
            cache_info = status.get('cache_status', {})
            
            if cache_info.get('enabled'):
                # Cache metrics overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    hit_rate = cache_info.get('hit_rate', 0)
                    st.metric("🎯 Cache Hit Rate", f"{hit_rate:.1%}")
                    
                    # Color-coded performance indicator
                    if hit_rate > 0.8:
                        st.success("🟢 Excellent performance")
                    elif hit_rate > 0.6:
                        st.info("🟡 Good performance")
                    elif hit_rate > 0.3:
                        st.warning("🟠 Moderate performance")
                    else:
                        st.error("🔴 Poor performance")
                
                with col2:
                    total_requests = cache_info.get('total_requests', 0)
                    st.metric("📊 Total Requests", f"{total_requests:,}")
                
                with col3:
                    cache_hits = cache_info.get('cache_hits', 0)
                    st.metric("✅ Cache Hits", f"{cache_hits:,}")
                
                with col4:
                    time_saved = cache_info.get('total_time_saved', 0)
                    st.metric("⚡ Time Saved", f"{time_saved:.1f}s")
                
                # Cache type and availability
                col1, col2 = st.columns(2)
                
                with col1:
                    cache_type = cache_info.get('cache_type', 'unknown')
                    redis_available = cache_info.get('redis_available', False)
                    
                    if redis_available:
                        st.success("🔥 Redis Cache: Production Ready")
                        st.write("• High-performance distributed caching")
                        st.write("• Persistent across system restarts")
                        st.write("• Optimized for < 200ms response times")
                    else:
                        st.info("💾 In-Memory Cache: Development Mode")
                        st.write("• Fast local caching")
                        st.write("• Automatic fallback system")
                        st.write("• Clears on system restart")
                
                with col2:
                    # Cache management controls
                    st.write("**🛠️ Cache Management:**")
                    
                    col_clear, col_warm = st.columns(2)
                    
                    with col_clear:
                        if st.button("🗑️ Clear Cache", help="Clear all cached data to free memory"):
                            with st.spinner("Clearing cache..."):
                                result = st.session_state.enhanced_rag.clear_cache()
                                
                                if result.get("success"):
                                    cleared = result.get("cleared_entries", 0)
                                    st.success(f"✅ Cleared {cleared:,} cache entries")
                                else:
                                    st.error(f"❌ Failed to clear cache: {result.get('error', 'Unknown error')}")
                    
                    with col_warm:
                        if st.button("🔥 Warm Cache", help="Pre-populate cache with common queries"):
                            with st.spinner("Warming cache..."):
                                # Common queries for cache warming
                                common_queries = [
                                    "What are the main topics?",
                                    "Summarize the key points",
                                    "What are the most important concepts?",
                                    "How do these topics relate?",
                                    "What are the conclusions?"
                                ]
                                
                                result = st.session_state.enhanced_rag.warm_cache(common_queries)
                                
                                if result.get("success"):
                                    created = result.get("cache_entries_created", 0)
                                    st.success(f"🔥 Warmed cache with {created} entries")
                                else:
                                    st.error(f"❌ Cache warming failed: {result.get('error', 'Unknown error')}")
                
                # Detailed cache analytics
                if st.button("📊 Detailed Cache Analytics"):
                    with st.spinner("Loading detailed cache analytics..."):
                        cache_stats_result = st.session_state.enhanced_rag.get_cache_stats()
                        
                        if cache_stats_result.get("success"):
                            cache_stats = cache_stats_result["stats"]
                            
                            st.subheader("📈 Detailed Cache Analytics")
                            
                            # Namespace breakdown
                            namespaces = cache_stats.get("namespaces", {})
                            if namespaces:
                                st.write("**🗂️ Cache Namespace Breakdown:**")
                                
                                for namespace, ns_stats in namespaces.items():
                                    with st.expander(f"📁 {namespace.title().replace('_', ' ')} ({ns_stats['count']} entries)"):
                                        col1_ns, col2_ns, col3_ns = st.columns(3)
                                        
                                        with col1_ns:
                                            st.metric("Entries", f"{ns_stats['count']:,}")
                                        with col2_ns:
                                            size_mb = ns_stats['total_size'] / (1024 * 1024)
                                            st.metric("Size", f"{size_mb:.2f} MB")
                                        with col3_ns:
                                            st.metric("Total Accesses", f"{ns_stats['total_accesses']:,}")
                                        
                                        if ns_stats['count'] > 0:
                                            avg_size = ns_stats['total_size'] / ns_stats['count']
                                            avg_accesses = ns_stats['total_accesses'] / ns_stats['count']
                                            
                                            st.write(f"• **Avg Entry Size:** {avg_size:.0f} bytes")
                                            st.write(f"• **Avg Accesses per Entry:** {avg_accesses:.1f}")
                            
                            # Redis-specific information
                            if cache_stats.get("redis_info"):
                                st.write("**🔥 Redis Server Information:**")
                                redis_info = cache_stats["redis_info"]
                                
                                col1_redis, col2_redis, col3_redis = st.columns(3)
                                
                                with col1_redis:
                                    st.metric("Memory Used", redis_info.get('used_memory_human', 'N/A'))
                                with col2_redis:
                                    st.metric("Connected Clients", redis_info.get('connected_clients', 0))
                                with col3_redis:
                                    keyspace_hits = redis_info.get('keyspace_hits', 0)
                                    keyspace_misses = redis_info.get('keyspace_misses', 0)
                                    total_keyspace = keyspace_hits + keyspace_misses
                                    
                                    if total_keyspace > 0:
                                        redis_hit_rate = keyspace_hits / total_keyspace
                                        st.metric("Redis Hit Rate", f"{redis_hit_rate:.1%}")
                                    else:
                                        st.metric("Redis Hit Rate", "N/A")
                        
                        else:
                            st.error(f"❌ Failed to load cache analytics: {cache_stats_result.get('error', 'Unknown error')}")
                
                # Performance impact visualization
                if cache_info.get('total_requests', 0) > 0:
                    st.write("**📊 Performance Impact:**")
                    
                    # Calculate performance improvements
                    hit_rate = cache_info.get('hit_rate', 0)
                    avg_retrieval = cache_info.get('avg_retrieval_time', 0)
                    
                    if hit_rate > 0 and avg_retrieval > 0:
                        # Estimate time savings
                        estimated_query_time = avg_retrieval / hit_rate if hit_rate > 0 else avg_retrieval
                        time_saved_per_hit = estimated_query_time * 0.9  # Assume 90% time saving
                        
                        total_potential_time = cache_info.get('total_requests', 0) * estimated_query_time
                        actual_time_spent = (cache_info.get('cache_misses', 0) * estimated_query_time + 
                                           cache_info.get('cache_hits', 0) * (estimated_query_time * 0.1))
                        
                        performance_improvement = ((total_potential_time - actual_time_spent) / 
                                                 total_potential_time) if total_potential_time > 0 else 0
                        
                        # Performance metrics
                        col1_perf, col2_perf = st.columns(2)
                        
                        with col1_perf:
                            st.metric("🚀 Performance Improvement", f"{performance_improvement:.1%}")
                            st.progress(performance_improvement, text=f"Speed boost: {performance_improvement:.1%}")
                        
                        with col2_perf:
                            target_rate = 0.2  # 200ms target
                            if avg_retrieval > 0:
                                target_achievement = min(target_rate / avg_retrieval, 1.0)
                                st.metric("🎯 Target Achievement", f"{target_achievement:.1%}")
                                st.progress(target_achievement, text=f"< 200ms goal: {target_achievement:.1%}")
            
            else:
                st.warning("⚠️ High-performance caching is disabled or unavailable")
        
        # System health monitoring
        st.divider()
        st.subheader("🏥 System Health Monitor")
        
        health_cols = st.columns(6)
        
        with health_cols[0]:
            if status.get('api_keys_available', False):
                st.success("🤖 LLM: Online")
            else:
                st.info("🔧 LLM: Offline")
        
        with health_cols[1]:
            if status.get('documents_processed', 0) > 0:
                st.success("📄 Docs: Ready")
            else:
                st.warning("📄 Docs: Empty")
        
        with health_cols[2]:
            if capabilities.get('question_answering'):
                st.success("💬 Q&A: Active")
            else:
                st.error("💬 Q&A: Inactive")
        
        with health_cols[3]:
            if capabilities.get('hybrid_search'):
                st.success("🎯 Search: Pro")
            else:
                st.warning("🎯 Search: Basic")
        
        with health_cols[4]:
            if capabilities.get('comprehensive_evaluation'):
                st.success("📊 Eval: Ready")
            else:
                st.warning("📊 Eval: Limited")
        
        with health_cols[5]:
            if capabilities.get('high_performance_caching'):
                cache_status = status.get('cache_status', {})
                if cache_status.get('redis_available', False):
                    st.success("🚀 Cache: Redis")
                elif cache_status.get('enabled', False):
                    st.info("💾 Cache: Memory")
                else:
                    st.warning("❌ Cache: Off")
            else:
                st.warning("❌ Cache: Disabled")
        
        # Advanced configuration and tuning
        with st.expander("⚙️ Advanced Configuration & Tuning"):
            st.write("**🔧 System Configuration:**")
            
            config_info = {
                "System Mode": status.get('mode', 'unknown'),
                "Chunk Size": st.session_state.config.CHUNK_SIZE,
                "Chunk Overlap": st.session_state.config.CHUNK_OVERLAP,
                "Temperature": st.session_state.config.TEMPERATURE,
                "Max Tokens": st.session_state.config.MAX_TOKENS,
                "Embedding Model": st.session_state.config.EMBEDDING_MODEL
            }
            
            # Display as metrics
            config_cols = st.columns(3)
            
            with config_cols[0]:
                st.metric("Chunk Size", f"{config_info['Chunk Size']} chars")
                st.metric("Chunk Overlap", f"{config_info['Chunk Overlap']} chars")
            
            with config_cols[1]:
                st.metric("Temperature", config_info['Temperature'])
                st.metric("Max Tokens", config_info['Max Tokens'])
            
            with config_cols[2]:
                st.write(f"**Model:** {config_info['Embedding Model']}")
                st.write(f"**Extensions:** {', '.join(st.session_state.config.SUPPORTED_EXTENSIONS)}")
            
        # Performance recommendations
        st.divider()
        st.subheader("💡 Performance Recommendations")
        
        recommendations = []
        
        # Generate recommendations based on current state
        if status.get('documents_processed', 0) == 0:
            recommendations.append("📄 Upload documents to enable full functionality")
        
        if not capabilities.get('hybrid_search'):
            recommendations.append("🎯 Enable hybrid search for better retrieval performance")
        
        if not capabilities.get('reranking'):
            recommendations.append("🎭 Consider enabling reranking for improved relevance")
        
        if st.session_state.chat_history:
            avg_time = sum(chat.get('response_time', 0) for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
            if avg_time > 2.0:
                recommendations.append("⚡ Response times are high - consider optimizing chunk size or model selection")
            elif avg_time < 0.2:
                recommendations.append("🚀 Excellent response times! System is well-optimized")
        
        if not capabilities.get('comprehensive_evaluation'):
            recommendations.append("📊 Enable evaluation framework to measure and improve performance")
        
        if not recommendations:
            recommendations.append("✅ System is running optimally with all pro features enabled!")
        
        for rec in recommendations:
            st.info(rec)
        
        # Real-time monitoring refresh
        if st.button("🔄 Refresh Dashboard", type="secondary"):
            st.rerun()
        
        # Export system report
        if st.button("📊 Export System Report"):
            import json
            from datetime import datetime
            
            system_report = {
                "timestamp": datetime.now().isoformat(),
                "system_status": status,
                "chat_history_count": len(st.session_state.chat_history),
                "response_times": [chat.get('response_time', 0) for chat in st.session_state.chat_history],
                "recommendations": recommendations
            }
            
            report_json = json.dumps(system_report, indent=2)
            st.download_button(
                label="💾 Download Report",
                data=report_json,
                file_name=f"rag_system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with tab9:
        st.header("🎨 Multi-Modal Elements Discovery")
        st.write("Explore tables, images, charts, and other visual elements extracted from your documents.")
        
        # Check if documents are loaded
        status = st.session_state.enhanced_rag.get_system_status()
        capabilities = status.get('capabilities', {})
        
        if status['documents_processed'] == 0:
            st.warning("⚠️ No documents loaded. Please upload and process documents first.")
        else:
            # Multi-modal capabilities overview
            st.subheader("🚀 Multi-Modal Capabilities")
            
            multimodal_info = status.get('multimodal_status', {})
            
            if capabilities.get('multimodal_processing'):
                # Capability badges
                capability_cols = st.columns(4)
                
                with capability_cols[0]:
                    table_status = "✅ Available" if capabilities.get('table_processing') else "❌ Limited"
                    st.metric("📊 Table Processing", table_status)
                
                with capability_cols[1]:
                    image_status = "✅ Available" if capabilities.get('image_processing') else "❌ Limited"
                    st.metric("🖼️ Image Processing", image_status)
                
                with capability_cols[2]:
                    ai_status = "✅ Available" if capabilities.get('ai_image_understanding') else "❌ Limited"
                    st.metric("🤖 AI Understanding", ai_status)
                
                with capability_cols[3]:
                    elements_count = multimodal_info.get('total_elements', 0)
                    st.metric("🎨 Elements Found", elements_count)
                
                # Multi-modal element summary
                if multimodal_info.get('enabled') and multimodal_info.get('total_elements', 0) > 0:
                    st.divider()
                    st.subheader("📈 Multi-Modal Elements Overview")
                    
                    element_types = multimodal_info.get('element_types', {})
                    capabilities_detail = multimodal_info.get('capabilities', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🎭 Element Types Found:**")
                        for element_type, count in element_types.items():
                            if count > 0:
                                type_emoji = {"table": "📊", "image": "🖼️", "chart": "📈", "diagram": "🔄"}.get(element_type, "🎨")
                                st.write(f"{type_emoji} **{element_type.title()}**: {count} elements")
                    
                    with col2:
                        st.write("**🛠️ Processing Capabilities:**")
                        for capability, available in capabilities_detail.items():
                            status_icon = "✅" if available else "❌"
                            capability_name = capability.replace('_', ' ').title()
                            st.write(f"{status_icon} {capability_name}")
                    
                    # Element search and exploration
                    st.divider()
                    st.subheader("🔍 Search Multi-Modal Elements")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        search_query = st.text_input(
                            "Search for specific elements:",
                            placeholder="e.g., 'revenue table', 'performance chart', 'diagram'",
                            key="multimodal_search"
                        )
                    
                    with col2:
                        element_filter = st.multiselect(
                            "Filter by type:",
                            ["table", "image", "chart", "diagram"],
                            default=[],
                            key="multimodal_filter"
                        )
                    
                    # Search execution
                    if st.button("🔍 Search Elements", type="primary") and search_query:
                        with st.spinner("Searching multi-modal elements..."):
                            search_result = st.session_state.enhanced_rag.query_multimodal_elements(
                                search_query, 
                                element_filter if element_filter else None
                            )
                            
                            if search_result.get("success"):
                                results = search_result["results"]
                                
                                if results:
                                    st.success(f"🎯 Found {len(results)} matching elements")
                                    
                                    # Display results
                                    for i, element in enumerate(results):
                                        element_type = element["element_type"]
                                        element_emoji = {"table": "📊", "image": "🖼️", "chart": "📈", "diagram": "🔄"}.get(element_type, "🎨")
                                        
                                        confidence = element["confidence_score"]
                                        confidence_color = "success" if confidence > 0.8 else "warning" if confidence > 0.6 else "error"
                                        
                                        with st.expander(f"{element_emoji} {element_type.title()} #{i+1} (Confidence: {confidence:.1%})"):
                                            st.write(f"**🆔 Element ID:** {element['element_id']}")
                                            st.write(f"**🔧 Processing Method:** {element['processing_method']}")
                                            
                                            # Element-specific information
                                            if element_type == "table" and "table_info" in element:
                                                table_info = element["table_info"]
                                                col1_t, col2_t, col3_t = st.columns(3)
                                                
                                                with col1_t:
                                                    st.metric("Rows", table_info["num_rows"])
                                                with col2_t:
                                                    st.metric("Columns", table_info["num_columns"])
                                                with col3_t:
                                                    st.metric("Data Points", table_info["num_rows"] * table_info["num_columns"])
                                                
                                                if table_info.get("columns"):
                                                    st.write("**📋 Column Names:**")
                                                    st.write(", ".join(table_info["columns"]))
                                            
                                            elif element_type in ["image", "chart"] and "image_info" in element:
                                                image_info = element["image_info"]
                                                col1_i, col2_i = st.columns(2)
                                                
                                                with col1_i:
                                                    st.write(f"**📐 Dimensions:** {image_info['dimensions']}")
                                                    st.write(f"**🎨 Format:** {image_info['format']}")
                                                
                                                with col2_i:
                                                    st.write(f"**🎯 Objects Detected:** {image_info['detected_objects']}")
                                                    has_text_icon = "✅" if image_info['has_text'] else "❌"
                                                    st.write(f"**📝 Contains Text:** {has_text_icon}")
                                            
                                            # Description
                                            st.write("**📝 AI Description:**")
                                            st.info(element["text_description"])
                                            
                                            # Show confidence indicator
                                            st.progress(confidence, text=f"Processing confidence: {confidence:.1%}")
                                
                                else:
                                    st.info("🔍 No elements found matching your search criteria.")
                            
                            else:
                                st.error(f"❌ Search failed: {search_result.get('error', 'Unknown error')}")
                    
                    # Export functionality
                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📊 Get Detailed Summary", help="Get comprehensive summary of all elements"):
                            with st.spinner("Generating detailed summary..."):
                                summary_result = st.session_state.enhanced_rag.get_multimodal_summary()
                                
                                if summary_result.get("success"):
                                    summary = summary_result["summary"]
                                    
                                    st.subheader("📈 Detailed Multi-Modal Summary")
                                    
                                    # Processing statistics
                                    col1_s, col2_s, col3_s = st.columns(3)
                                    
                                    with col1_s:
                                        st.metric("Total Elements", summary["total_elements"])
                                    with col2_s:
                                        st.metric("Avg Confidence", f"{summary['average_confidence']:.1%}")
                                    with col3_s:
                                        st.metric("With Structured Data", summary["elements_with_structured_data"])
                                    
                                    # Processing methods breakdown
                                    if summary.get("processing_methods"):
                                        st.write("**🔧 Processing Methods Used:**")
                                        methods = summary["processing_methods"]
                                        for method, count in methods.items():
                                            pct = count / summary["total_elements"] if summary["total_elements"] > 0 else 0
                                            st.progress(pct, text=f"{method.replace('_', ' ').title()}: {count} ({pct:.1%})")
                                
                                else:
                                    st.error(f"❌ Summary failed: {summary_result.get('error', 'Unknown error')}")
                    
                    with col2:
                        if st.button("💾 Export Elements", help="Export all multi-modal elements to JSON"):
                            with st.spinner("Exporting multi-modal elements..."):
                                import tempfile
                                
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                                    export_result = st.session_state.enhanced_rag.export_multimodal_elements(tmp_file.name)
                                    
                                    if export_result.get("success"):
                                        # Read the file for download
                                        with open(tmp_file.name, 'r') as f:
                                            export_data = f.read()
                                        
                                        st.download_button(
                                            label="📥 Download Elements Export",
                                            data=export_data,
                                            file_name=f"multimodal_elements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            mime="application/json"
                                        )
                                        
                                        st.success("✅ Elements exported successfully!")
                                    
                                    else:
                                        st.error(f"❌ Export failed: {export_result.get('error', 'Unknown error')}")
                    
                    # Tips for multi-modal usage
                    with st.expander("💡 Multi-Modal Usage Tips"):
                        st.markdown("""
                        **🎯 Best Practices for Multi-Modal RAG:**
                        
                        **📊 Table Processing:**
                        - Tables are automatically detected in HTML and text formats
                        - Use specific column names in your queries for better results  
                        - Ask about data relationships, trends, and statistics
                        
                        **🖼️ Image Analysis:**
                        - Images are analyzed for content, text, and objects
                        - AI generates descriptions of visual content
                        - OCR extracts any text found in images
                        
                        **📈 Chart Understanding:**
                        - Charts are automatically classified by type
                        - Query about data trends, comparisons, and insights
                        - Text and numeric data from charts is extracted
                        
                        **🔍 Search Tips:**
                        - Use descriptive terms: "revenue table", "performance chart"
                        - Combine element types in queries: "sales data visualization"
                        - Ask about specific data points or trends shown in elements
                        """)
                
                else:
                    st.info("📝 No multi-modal elements found yet. Upload documents with tables, images, or charts to get started!")
            
            else:
                st.warning("⚠️ Multi-modal processing is not enabled or dependencies are missing.")
                
                st.subheader("📦 Required Dependencies")
                dependencies_info = {
                    "pandas": "Table processing and data analysis",
                    "Pillow (PIL)": "Image processing and manipulation", 
                    "opencv-python": "Advanced image analysis and chart detection",
                    "pytesseract": "OCR text extraction from images",
                    "transformers": "AI-powered image understanding and object detection"
                }
                
                for dep, description in dependencies_info.items():
                    st.write(f"• **{dep}**: {description}")
                
                st.info("💡 Install missing dependencies to enable full multi-modal capabilities!")
    
    with tab10:
        st.header("🕸️ Knowledge Graph Explorer")
        st.write("Discover entity relationships and semantic connections in your documents through an intelligent knowledge graph.")
        
        # Check if documents are loaded
        status = st.session_state.enhanced_rag.get_system_status()
        capabilities = status.get('capabilities', {})
        
        if status['documents_processed'] == 0:
            st.warning("⚠️ No documents loaded. Please upload and process documents first.")
        else:
            # Knowledge graph capabilities overview
            st.subheader("🧠 Graph Capabilities")
            
            graph_info = status.get('knowledge_graph', {})
            
            if capabilities.get('graph_enhanced_rag'):
                # Capability metrics
                graph_cols = st.columns(4)
                
                with graph_cols[0]:
                    entities_count = graph_info.get('entities_extracted', 0)
                    st.metric("👥 Entities", entities_count)
                
                with graph_cols[1]:
                    relations_count = graph_info.get('relations_extracted', 0)
                    st.metric("🔗 Relations", relations_count)
                
                with graph_cols[2]:
                    processing_time = graph_info.get('processing_time', 0)
                    st.metric("⏱️ Build Time", f"{processing_time:.2f}s")
                
                with graph_cols[3]:
                    graph_enabled = "✅ Enabled" if graph_info.get('enabled') else "❌ Disabled"
                    st.metric("🚀 Status", graph_enabled)
                
                if graph_info.get('enabled') and entities_count > 0:
                    st.divider()
                    
                    # Graph statistics
                    st.subheader("📈 Knowledge Graph Statistics")
                    
                    try:
                        stats_result = st.session_state.enhanced_rag.get_knowledge_graph_stats()
                        
                        if stats_result.get('success'):
                            stats = stats_result['statistics']
                            
                            # Main statistics
                            stats_cols = st.columns(3)
                            
                            with stats_cols[0]:
                                st.metric("🔢 Total Nodes", stats.get('num_nodes', 0))
                                st.metric("↔️ Total Edges", stats.get('num_edges', 0))
                            
                            with stats_cols[1]:
                                avg_degree = stats.get('average_degree', 0)
                                st.metric("📊 Avg Degree", f"{avg_degree:.2f}")
                                density = stats.get('graph_density', 0)
                                st.metric("🌐 Graph Density", f"{density:.4f}")
                            
                            with stats_cols[2]:
                                components = stats.get('connected_components', 0)
                                st.metric("🧩 Components", components)
                            
                            # Entity types breakdown
                            if stats.get('entity_types'):
                                st.subheader("🏷️ Entity Types")
                                entity_types = stats['entity_types']
                                
                                type_cols = st.columns(min(4, len(entity_types)))
                                for i, (entity_type, count) in enumerate(entity_types.items()):
                                    with type_cols[i % 4]:
                                        st.metric(f"{entity_type}", count)
                            
                            # Relation types breakdown
                            if stats.get('relation_types'):
                                st.subheader("🔗 Relation Types")
                                relation_types = stats['relation_types']
                                
                                relation_cols = st.columns(min(3, len(relation_types)))
                                for i, (relation_type, count) in enumerate(relation_types.items()):
                                    with relation_cols[i % 3]:
                                        st.metric(relation_type.replace('_', ' '), count)
                            
                            # Most connected entities
                            if stats.get('most_connected_entities'):
                                st.subheader("⭐ Most Connected Entities")
                                
                                connected_entities = stats['most_connected_entities'][:10]
                                for i, entity_info in enumerate(connected_entities):
                                    entity_name = entity_info.get('name', 'Unknown')
                                    centrality = entity_info.get('centrality', 0)
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"**{i+1}. {entity_name}**")
                                    with col2:
                                        st.write(f"Score: {centrality:.3f}")
                        
                        else:
                            st.error(f"Failed to get graph statistics: {stats_result.get('error')}")
                    
                    except Exception as e:
                        st.error(f"Error loading graph statistics: {e}")
                    
                    st.divider()
                    
                    # Knowledge graph query interface
                    st.subheader("🔍 Graph Query Explorer")
                    
                    query_cols = st.columns([3, 1])
                    
                    with query_cols[0]:
                        graph_query = st.text_input(
                            "Enter your query",
                            placeholder="Ask about entities, relationships, or concepts...",
                            help="Query the knowledge graph to find entity relationships and semantic connections"
                        )
                    
                    with query_cols[1]:
                        top_k = st.number_input("Top Results", min_value=1, max_value=20, value=10)
                    
                    if st.button("🔍 Query Knowledge Graph", type="primary"):
                        if graph_query:
                            with st.spinner("🕸️ Searching knowledge graph..."):
                                try:
                                    result = st.session_state.enhanced_rag.query_knowledge_graph(graph_query, top_k)
                                    
                                    if result.get('success'):
                                        graph_results = result['graph_results']
                                        
                                        st.success(f"✅ Found insights for: **{graph_query}**")
                                        
                                        # Display direct matches
                                        direct_matches = graph_results.get('direct_matches', [])
                                        if direct_matches:
                                            st.subheader("🎯 Direct Entity Matches")
                                            
                                            for i, match in enumerate(direct_matches):
                                                entity_info = match.get('entity', {})
                                                related_entities = match.get('related_entities', [])
                                                
                                                with st.expander(f"📍 {entity_info.get('name', 'Unknown Entity')} ({entity_info.get('entity_type', 'Unknown')})"):
                                                    st.write(f"**Description:** {entity_info.get('text_description', 'No description available')}")
                                                    st.write(f"**Confidence:** {entity_info.get('confidence', 0):.2f}")
                                                    
                                                    if related_entities:
                                                        st.write("**Related Entities:**")
                                                        for related in related_entities[:5]:  # Show top 5
                                                            related_entity = related.get('entity', {})
                                                            connection_path = related.get('connection_path', [])
                                                            
                                                            st.write(f"• {related_entity.get('name', 'Unknown')} (distance: {len(connection_path)})")
                                        
                                        # Display semantic expansion
                                        semantic_expansion = graph_results.get('semantic_expansion', [])
                                        if semantic_expansion:
                                            st.subheader("🌐 Semantic Expansion")
                                            
                                            for i, expansion in enumerate(semantic_expansion[:5]):
                                                entity_info = expansion.get('entity', {})
                                                relevance_score = expansion.get('relevance_score', 0)
                                                connection_path = expansion.get('connection_path', [])
                                                
                                                st.write(f"**{entity_info.get('name', 'Unknown')}** (relevance: {relevance_score:.3f}, hops: {len(connection_path)})")
                                                st.write(f"   {entity_info.get('text_description', 'No description')[:150]}...")
                                        
                                        # Query analysis
                                        query_analysis = graph_results.get('query_analysis', {})
                                        if query_analysis:
                                            st.subheader("🔬 Query Analysis")
                                            st.write(f"**Entities Found in Query:** {query_analysis.get('entities_found', 0)}")
                                            
                                            query_entities = query_analysis.get('query_entities', [])
                                            if query_entities:
                                                st.write("**Detected Entities:**")
                                                for entity in query_entities:
                                                    st.write(f"• {entity.get('name', 'Unknown')} ({entity.get('entity_type', 'Unknown')})")
                                    
                                    else:
                                        st.error(f"Query failed: {result.get('error')}")
                                
                                except Exception as e:
                                    st.error(f"Knowledge graph query error: {e}")
                        else:
                            st.warning("Please enter a query")
                    
                    st.divider()
                    
                    # Entity context explorer
                    st.subheader("👤 Entity Context Explorer")
                    
                    entity_name = st.text_input(
                        "Entity Name",
                        placeholder="Enter entity name to explore its context...",
                        help="Get detailed context and relationships for a specific entity"
                    )
                    
                    if st.button("🔍 Explore Entity Context"):
                        if entity_name:
                            with st.spinner(f"🕸️ Loading context for {entity_name}..."):
                                try:
                                    context_result = st.session_state.enhanced_rag.get_entity_context(entity_name)
                                    
                                    if context_result.get('success'):
                                        context = context_result['context']
                                        
                                        if 'error' not in context:
                                            entity = context.get('entity', {})
                                            
                                            st.success(f"✅ Found context for: **{entity_name}**")
                                            
                                            # Entity details
                                            st.subheader("📋 Entity Details")
                                            
                                            details_cols = st.columns(3)
                                            with details_cols[0]:
                                                st.write(f"**Name:** {entity.get('name', 'Unknown')}")
                                                st.write(f"**Type:** {entity.get('entity_type', 'Unknown')}")
                                            
                                            with details_cols[1]:
                                                st.write(f"**Confidence:** {entity.get('confidence', 0):.2f}")
                                                aliases = entity.get('aliases', [])
                                                st.write(f"**Aliases:** {len(aliases)}")
                                            
                                            with details_cols[2]:
                                                mentions = entity.get('mentions', [])
                                                st.write(f"**Mentions:** {len(mentions)}")
                                            
                                            # Context summary
                                            context_summary = context.get('context_summary', '')
                                            if context_summary:
                                                st.subheader("📝 Context Summary")
                                                st.write(context_summary)
                                            
                                            # Direct relations
                                            direct_relations = context.get('direct_relations', [])
                                            if direct_relations:
                                                st.subheader("🔗 Direct Relations")
                                                
                                                for relation in direct_relations[:10]:  # Show top 10
                                                    rel_type = relation.get('relation_type', 'Unknown')
                                                    confidence = relation.get('confidence', 0)
                                                    
                                                    st.write(f"• **{rel_type}** (confidence: {confidence:.2f})")
                                            
                                            # Related entities
                                            related_entities = context.get('related_entities', [])
                                            if related_entities:
                                                st.subheader("🌐 Related Entities")
                                                
                                                for related in related_entities[:10]:  # Show top 10
                                                    related_entity = related.get('entity', {})
                                                    distance = related.get('distance', 0)
                                                    
                                                    st.write(f"• **{related_entity.get('name', 'Unknown')}** ({related_entity.get('entity_type', 'Unknown')}) - distance: {distance}")
                                        
                                        else:
                                            st.error(f"Entity not found: {context['error']}")
                                    else:
                                        st.error(f"Context retrieval failed: {context_result.get('error')}")
                                
                                except Exception as e:
                                    st.error(f"Entity context error: {e}")
                        else:
                            st.warning("Please enter an entity name")
                    
                    st.divider()
                    
                    # Export options
                    st.subheader("💾 Export Knowledge Graph")
                    
                    export_cols = st.columns(3)
                    
                    with export_cols[0]:
                        export_format = st.selectbox(
                            "Export Format",
                            ["json", "gexf", "pickle"],
                            help="Choose export format: JSON (readable), GEXF (Gephi), Pickle (Python)"
                        )
                    
                    with export_cols[1]:
                        if st.button("📁 Export Graph"):
                            try:
                                from datetime import datetime
                                filename = f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                                filepath = f"exports/{filename}"
                                
                                # Create exports directory if it doesn't exist
                                import os
                                os.makedirs("exports", exist_ok=True)
                                
                                result = st.session_state.enhanced_rag.export_knowledge_graph(filepath, export_format)
                                
                                if result.get('success'):
                                    st.success(f"✅ Graph exported to: {filename}")
                                    
                                    # Offer download
                                    if os.path.exists(filepath):
                                        with open(filepath, 'rb') as f:
                                            file_data = f.read()
                                        
                                        st.download_button(
                                            label="💾 Download Export",
                                            data=file_data,
                                            file_name=filename,
                                            mime="application/octet-stream"
                                        )
                                else:
                                    st.error(f"Export failed: {result.get('error')}")
                            
                            except Exception as e:
                                st.error(f"Export error: {e}")
                
                else:
                    st.info("ℹ️ Knowledge graph is empty. Process some documents to build the graph.")
            
            else:
                st.warning("⚠️ Knowledge graph processing is not enabled or dependencies are missing.")
                
                st.subheader("📦 Required Dependencies")
                dependencies_info = {
                    "spacy": "Named entity recognition and dependency parsing",
                    "networkx": "Graph data structure and algorithms",
                    "sentence-transformers": "Entity embeddings and semantic similarity (optional)",
                    "scikit-learn": "Advanced clustering and similarity metrics (optional)"
                }
                
                for dep, description in dependencies_info.items():
                    st.write(f"• **{dep}**: {description}")
                
                st.info("💡 Install missing dependencies and run 'python -m spacy download en_core_web_sm' to enable knowledge graph features!")

if __name__ == "__main__":
    main()