#!/bin/bash

# Check Python version and install if needed
check_python() {
    if command -v python3.10 &> /dev/null; then
        echo "Python 3.10 found"
        return 0
    fi
    
    echo "Python 3.10 not found. Installing..."
    if [ "$(uname)" = "Darwin" ]; then
        brew install python@3.10
    elif [ -f /etc/debian_version ]; then
        sudo apt-get update
        sudo apt-get install -y python3.10 python3.10-venv
    elif [ -f /etc/redhat-release ]; then
        sudo dnf install -y python3.10
    fi
}

# Ensure Python 3.10 is available
check_python

# Create virtual environment with Python 3.10
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install requirements
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo ""
    echo "WARNING: Ollama not found. Please install from https://ollama.ai"
    echo "After installing, run: ollama pull llama3.1:8b && ollama pull nomic-embed-text"
fi

echo "Setup complete! To start using the application:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "2. Run the application:"
echo "   python main.py"
