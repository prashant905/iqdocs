#!/bin/bash

echo "🚀 Setting up File Upload & OpenAI Analysis App"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp env.example .env
    echo "✏️  Please edit the .env file and add your OpenAI API key!"
else
    echo "📄 .env file already exists"
fi

echo "✅ Setup complete!"
echo ""
echo "🔑 IMPORTANT: Add your OpenAI API key to the .env file:"
echo "   Edit .env and set: OPENAI_API_KEY=your-actual-api-key"
echo ""
echo "🔑 Alternative: Set as environment variable:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "🏃 To run the app:"
echo "source venv/bin/activate"
echo "python app.py"
echo ""
echo "🌐 The app will be available at: http://localhost:7860" 