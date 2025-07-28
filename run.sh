#!/bin/bash

# RAG Document Q&A System Startup Script

echo "🚀 Starting RAG Document Q&A System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/installed.flag" ]; then
    echo "📋 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/installed.flag
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo "🔑 Please edit .env file and add your API keys:"
    echo "   - OPENAI_API_KEY=your_key_here"
    echo "   - ANTHROPIC_API_KEY=your_key_here"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if API keys are configured
if ! grep -q "^OPENAI_API_KEY=sk-" .env && ! grep -q "^ANTHROPIC_API_KEY=" .env; then
    echo "⚠️  No API keys found in .env file."
    echo "🔑 Please add at least one API key to .env file:"
    echo "   - OPENAI_API_KEY=your_key_here"
    echo "   - ANTHROPIC_API_KEY=your_key_here"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Create directories
mkdir -p uploads vector_store

echo "✅ Setup complete!"
echo "🌐 Starting Streamlit application..."
echo "📱 The app will open in your browser at: http://localhost:8501"
echo ""

# Start the application
streamlit run app.py