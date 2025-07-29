#!/usr/bin/env python3
"""
Demo script to showcase the enhanced model management interface
Run this to see the new radio button interface and progress tracking in action
"""

import streamlit as st
import time
from src.model_manager import ModelManager, ModelDownloadProgress

def main():
    st.title("🤖 Enhanced Model Manager Demo")
    st.write("This demo showcases the new radio button interface and download progress tracking")
    
    # Initialize model manager
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    model_manager = st.session_state.model_manager
    
    # Show available models with radio buttons
    st.header("📱 Radio Button Interface")
    st.write("**Question Answering Models:**")
    
    qa_models = model_manager.get_models_by_task("question-answering")
    qa_options = []
    qa_keys = []
    
    for key, model_info in qa_models.items():
        status_icon = "✅" if model_info.is_downloaded else "📥"
        size_text = f"({model_info.size_mb}MB)"
        performance = f"★{model_info.performance_score:.1f}"
        memory = model_info.memory_usage
        option_text = f"{status_icon} {model_info.name} {size_text} {performance} - {memory}"
        qa_options.append(option_text)
        qa_keys.append(key)
    
    if qa_options:
        selected_qa_index = st.radio(
            "Choose your QA model:",
            range(len(qa_options)),
            format_func=lambda x: qa_options[x],
            key="demo_qa_selection",
            help="Select a question-answering model. ✅ = Downloaded, 📥 = Needs download"
        )
        selected_qa_key = qa_keys[selected_qa_index]
        
        st.write(f"**Selected:** {selected_qa_key}")
        
        # Show model details
        selected_model = model_manager.get_model_info(selected_qa_key)
        if selected_model:
            with st.expander("📋 Model Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Size", f"{selected_model.size_mb}MB")
                with col2:
                    st.metric("Performance", f"★{selected_model.performance_score:.1f}")
                with col3:
                    st.metric("Memory", selected_model.memory_usage)
                
                st.write(f"**Description:** {selected_model.description}")
                st.write(f"**Task:** {selected_model.task}")
                st.write(f"**Languages:** {', '.join(selected_model.supported_languages)}")
    
    # Demonstrate progress tracking
    st.header("📊 Download Progress Demo")
    
    if st.button("🎬 Demo Download Progress", type="primary"):
        st.write("**Simulating model download with progress tracking...**")
        
        # Create progress containers
        progress_container = st.container()
        with progress_container:
            st.write("**DistilBERT QA Model**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate download progress
            total_size = 250 * 1024 * 1024  # 250MB in bytes
            progress = ModelDownloadProgress("demo-model", total_size)
            
            # Simulate progressive updates
            steps = ["Initializing...", "Downloading tokenizer...", "Downloading model...", 
                    "Saving locally...", "Finalizing...", "Complete!"]
            
            for i, step in enumerate(steps):
                progress.status = "downloading" if i < len(steps) - 1 else "completed"
                progress.downloaded_size = int((i / (len(steps) - 1)) * total_size)
                
                percentage = progress.get_progress_percentage()
                speed = progress.get_speed_mbps()
                eta = progress.get_eta_seconds()
                
                # Update UI
                progress_bar.progress(percentage / 100.0)
                
                if i < len(steps) - 1:
                    status_msg = f"{step} {percentage:.1f}% ({speed:.1f} MB/s, ETA: {eta:.0f}s)"
                else:
                    status_msg = "✅ Download completed!"
                
                status_text.write(status_msg)
                time.sleep(1)  # Simulate download time
        
        st.success("🎉 Demo completed! This shows how real downloads will look.")
    
    # Model presets demo
    st.header("🎯 Model Presets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**⚡ Lightweight**")
        lightweight = model_manager.get_lightweight_models()
        total_size = model_manager.estimate_total_download_size(lightweight)
        st.write(f"Total size: {total_size}MB")
        for model_key in lightweight:
            model_info = model_manager.get_model_info(model_key)
            if model_info:
                st.write(f"• {model_info.name}")
    
    with col2:
        st.write("**🎯 Recommended**")
        recommended = model_manager.get_recommended_models()
        total_size = model_manager.estimate_total_download_size(recommended)
        st.write(f"Total size: {total_size}MB")
        for model_key in recommended:
            model_info = model_manager.get_model_info(model_key)
            if model_info:
                st.write(f"• {model_info.name}")
    
    with col3:
        st.write("**🚀 High Performance**")
        high_perf = model_manager.get_high_performance_models()
        total_size = model_manager.estimate_total_download_size(high_perf)
        st.write(f"Total size: {total_size}MB")
        for model_key in high_perf:
            model_info = model_manager.get_model_info(model_key)
            if model_info:
                st.write(f"• {model_info.name}")
    
    # Storage information
    st.header("💾 Storage Information")
    storage_info = model_manager.get_storage_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📥 Downloaded Models", storage_info["downloaded_models"])
    with col2:
        st.metric("💾 Storage Used", f"{storage_info['total_size_mb']:.0f}MB")
    with col3:
        st.metric("📁 Models Directory", storage_info["models_dir"])
    
    # Key improvements summary
    st.header("✨ Key Improvements")
    
    improvements = [
        "🔘 **Radio buttons** instead of dropdown for better UX",
        "📊 **Real-time progress tracking** with speed and ETA",
        "📱 **Visual status indicators** (✅ downloaded, 📥 needs download)",
        "⭐ **Performance ratings** for each model",
        "💾 **Memory usage** information",
        "🎯 **Preset configurations** for different use cases",
        "📈 **Progressive download updates** throughout the process",
        "🔧 **Automatic model integration** after download"
    ]
    
    for improvement in improvements:
        st.write(improvement)
    
    st.success("🎉 Enhanced model management system is ready!")

if __name__ == "__main__":
    main()