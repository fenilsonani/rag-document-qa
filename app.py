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
    st.title("📚 RAG Document Q&A System")
    st.markdown("Ask questions about your documents using advanced AI retrieval and generation.")
    
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload Documents", 
        "💬 Smart Q&A", 
        "🔍 Search & Explore", 
        "🧠 Document Intelligence", 
        "🔗 Cross-References",
        "📊 Analytics Dashboard"
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
                # Show query enhancement if available
                try:
                    with st.expander("🔍 Query Analysis & Enhancement"):
                        st.info("Query analysis and enhancement features are available when documents are processed with intelligence engines.")
                except Exception as e:
                    pass
                
                with st.spinner("Generating intelligent answer..."):
                    result = st.session_state.enhanced_rag.ask_question(
                        question, 
                        use_conversation=use_conversation
                    )
                    
                    if result["success"]:
                        # Display answer
                        st.subheader("💡 Answer")
                        st.write(result["answer"])
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Response Time", f"{result['response_time']}s")
                        with col2:
                            st.metric("Sources Used", result['source_count'])
                        with col3:
                            st.metric("Conversation", "Yes" if use_conversation else "No")
                        
                        # Display sources
                        if result["sources"]:
                            display_sources(result["sources"])
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"],
                            "sources": len(result["sources"]),
                            "response_time": result["response_time"]
                        })
                    
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
        st.header("Search Documents")
        st.write("Search through your documents to find relevant content without generating answers.")
        
        search_query = st.text_input(
            "Search query:",
            placeholder="Enter keywords or phrases to search for...",
            key="search_input"
        )
        
        search_k = st.slider("Number of results", min_value=1, max_value=10, value=4)
        
        if st.button("🔍 Search", type="primary") and search_query:
            with st.spinner("Searching documents..."):
                result = st.session_state.enhanced_rag.search_documents(search_query, search_k)
                
                if result["success"]:
                    st.success(f"Found {result['result_count']} relevant chunks")
                    
                    for i, doc_result in enumerate(result["results"]):
                        with st.expander(f"Result {i+1}: {doc_result['metadata'].get('filename', 'Unknown')} (Score: {doc_result.get('similarity_score', 'N/A'):.3f})"):
                            st.write("**Content:**")
                            st.write(doc_result["content"])
                            
                            st.write("**Metadata:**")
                            st.json(doc_result["metadata"])
                
                else:
                    st.error(f"❌ Search failed: {result['error']}")

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
        st.header("📊 Advanced Analytics Dashboard")
        
        # Analytics Overview
        col1, col2, col3, col4 = st.columns(4)
        
        # System metrics
        status = st.session_state.enhanced_rag.get_system_status()
        
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
        
        # Performance Analytics
        if st.session_state.chat_history:
            st.subheader("⚡ Performance Analytics")
            
            # Calculate average response time
            response_times = [chat.get('response_time', 0) for chat in st.session_state.chat_history if 'response_time' in chat]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
                with col2:
                    st.metric("Fastest Response", f"{min(response_times):.2f}s")
                with col3:
                    st.metric("Slowest Response", f"{max(response_times):.2f}s")
                
                # Response time trend
                if len(response_times) > 1:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(range(len(response_times)), response_times, marker='o')
                    ax.set_title('Response Time Trend')
                    ax.set_xlabel('Question Number')
                    ax.set_ylabel('Response Time (seconds)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        # Query Analytics
        if st.session_state.chat_history:
            st.subheader("🔍 Query Analytics")
            
            # Analyze question patterns
            questions = [chat['question'] for chat in st.session_state.chat_history]
            if questions:
                st.write("**📊 Recent Questions:**")
                for i, question in enumerate(questions[-5:], 1):
                    st.write(f"{i}. {question[:100]}{'...' if len(question) > 100 else ''}")
        else:
            st.subheader("🔍 Query Analytics")
            st.info("Ask some questions to see analytics here!")
        
        # Document Intelligence Summary
        if st.session_state.document_insights:
            st.subheader("🧠 Intelligence Summary")
            insights = st.session_state.document_insights
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Quality metrics
                quality = insights.get('quality_assessment', {})
                if quality and not isinstance(quality, str):
                    st.write("**📊 Document Quality:**")
                    score = quality.get('overall_score', 0)
                    st.progress(score / 100)
                    st.write(f"Score: {score}/100 ({quality.get('quality_level', 'Unknown')})")
                
                # Complexity metrics
                complexity = insights.get('complexity_analysis', {})
                if complexity and not isinstance(complexity, str):
                    st.write("**📈 Complexity Analysis:**")
                    st.write(f"Level: {complexity.get('complexity_level', 'Unknown')}")
                    st.write(f"Flesch Score: {complexity.get('flesch_score', 0)}")
            
            with col2:
                # Content statistics
                stats = insights.get('document_statistics', {})
                if stats:
                    st.write("**📚 Content Statistics:**")
                    st.write(f"Total Words: {stats.get('total_words', 0):,}")
                    st.write(f"Unique Words: {stats.get('unique_words', 0):,}")
                    st.write(f"Vocabulary Richness: {stats.get('vocabulary_richness', 0):.1%}")
        
        # Cross-Reference Analytics
        if st.session_state.relationship_analysis:
            st.subheader("🔗 Relationship Analytics")
            analysis = st.session_state.relationship_analysis
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                similarities = analysis.get('similarity_matrix', {})
                if similarities:
                    pairs_count = len(similarities.get('high_similarity_pairs', []))
                    avg_sim = similarities.get('average_similarity', 0)
                    st.metric("Similar Pairs", pairs_count)
                    st.metric("Avg Similarity", f"{avg_sim:.2%}")
            
            with col2:
                contradictions = analysis.get('contradictions', [])
                agreements = analysis.get('agreements', [])
                st.metric("Contradictions", len(contradictions))
                st.metric("Agreements", len(agreements))
            
            with col3:
                graph = analysis.get('relationship_graph', {})
                if graph:
                    density = graph.get('network_density', 0)
                    relationships = graph.get('total_relationships', 0)
                    st.metric("Network Density", f"{density:.2%}")
                    st.metric("Total Relations", relationships)
        
        # System Health
        st.subheader("🏥 System Health")
        
        health_cols = st.columns(4)
        
        with health_cols[0]:
            if status.get('api_keys_available', False):
                st.success("🤖 LLM: Online")
            else:
                st.info("🔧 LLM: Offline Mode")
        
        with health_cols[1]:
            if status.get('documents_processed', 0) > 0:
                st.success("🗜️ Documents: Ready")
            else:
                st.warning("🗜️ Documents: Empty")
        
        with health_cols[2]:
            capabilities = status.get('capabilities', {})
            if capabilities.get('question_answering', False):
                st.success("🔗 Q&A: Active")
            else:
                st.error("🔗 Q&A: Inactive")
        
        with health_cols[3]:
            intelligence = status.get('capabilities', {}).get('document_intelligence', False)
            if intelligence:
                st.success("🧠 Intelligence: Active")
            else:
                st.warning("🧠 Intelligence: Limited")
        
        # Advanced Configuration
        with st.expander("⚙️ Advanced Configuration & System Details"):
            st.write("**Current Configuration:**")
            config_info = {
                "System Mode": status.get('mode', 'unknown'),
                "OpenAI Model": st.session_state.config.OPENAI_MODEL,
                "Anthropic Model": st.session_state.config.ANTHROPIC_MODEL,
                "Chunk Size": st.session_state.config.CHUNK_SIZE,
                "Chunk Overlap": st.session_state.config.CHUNK_OVERLAP,
                "Temperature": st.session_state.config.TEMPERATURE,
                "Max Tokens": st.session_state.config.MAX_TOKENS,
                "Embedding Model": st.session_state.config.EMBEDDING_MODEL
            }
            st.json(config_info)
            
            st.write("**Full System Status:**")
            st.json(status)
        
        if st.button("🔄 Refresh Analytics", type="secondary"):
            st.rerun()
        
        # Final system overview  
        st.divider()
        st.subheader("🏁 System Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Current Session:**")
            st.write(f"• Mode: {status.get('mode', 'unknown').title()}")
            st.write(f"• Documents: {status.get('documents_processed', 0)}")
            st.write(f"• Questions: {len(st.session_state.chat_history)}")
            st.write(f"• Suggestions: {len(st.session_state.get('smart_suggestions', []))}")
        
        with col2:
            st.write("**⚡ Performance:**")
            if st.session_state.chat_history:
                avg_time = sum(chat.get('response_time', 0) for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
                st.write(f"• Avg Response: {avg_time:.2f}s")
            else:
                st.write("• No questions asked yet")
        
        # Configuration
        with st.expander("⚙️ Configuration Details"):
            config_info = {
                "Chunk Size": st.session_state.config.CHUNK_SIZE,
                "Chunk Overlap": st.session_state.config.CHUNK_OVERLAP,
                "Temperature": st.session_state.config.TEMPERATURE,
                "Max Tokens": st.session_state.config.MAX_TOKENS,
                "Embedding Model": st.session_state.config.EMBEDDING_MODEL,
                "Supported Extensions": ", ".join(st.session_state.config.SUPPORTED_EXTENSIONS)
            }
            st.json(config_info)
        
        # Full system status
        with st.expander("📊 Full System Status (Advanced)"):
            st.json(status)

if __name__ == "__main__":
    main()