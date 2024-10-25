# USCIS Chatbot

An intelligent chatbot system designed to answer questions based on the USCIS manual using advanced language models and retrieval-augmented generation.

## Features

- Interactive interface using Streamlit
- RAG-based document retrieval using FAISS
- Intelligent context-aware responses using LLama 3.1 8b
- Citation support with page references

## Prerequisites

- Python 3.10 (required for compatibility)
- [Ollama](https://ollama.ai/) installed and running
- At least 16GB RAM recommended
- For fine-tuning (optional):
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon (M1/M2/M3) for MPS acceleration

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/uscis-chatbot.git
   cd uscis-chatbot
   ```

2. Run the setup script:
   
   On Unix/macOS (MacOS requires `brew` to be installed):
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   
   On Windows:
   ```bash
   ./setup.bat
   ```

3. Activate the virtual environment:
   
   On Unix/macOS:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   .\venv\Scripts\activate
   ```

4. Download the LLama model:
   ```bash
   ollama pull llama3.1:8b
   ```

## Running the Application

1. Ensure your virtual environment is activated

2. Start the application:
   ```bash
   python main.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

## Optional: Fine-tuning Setup (a trained model from this process is already included in the repo)

If you plan to use the fine-tuning functionality:

1. Ensure you are logged into Hugging Face through the Hugging Face CLI (`huggingface-cli login`) with an account that has access to the `meta-llama/Meta-Llama-3.2` model

2. Prepare your training data in the `data` directory (default is `data/uscis_manual.pdf`)

3. Run the fine-tuning script:
   ```bash
   python -m src.model.fine_tuning
   ```

Note: Fine-tuning requires significant computational resources. A GPU is recommended for reasonable training times.

## Technology Stack

- **Python**: Core programming language (3.8+)
- **LLama 3.1**: 8B parameter language model for natural language understanding
- **LangChain**: Framework for composing language model applications
- **FAISS**: Vector database for similarity search and document retrieval
- **Streamlit**: Web framework for building the user interface
- **Ollama**: LLM model serving and management platform

## Project Structure

```
uscis-chatbot/
├── data/              # Source document and FAISS vector store
├── config/
│   └── default.yaml   # Configuration settings
├── src/
│   ├── data/          # Data loading and processing
│   ├── interface/     # Streamlit UI components
│   ├── model/         # Model training and fine-tuning
│   ├── retrieval/     # Vector store and retrieval logic
│   └── utils/         # Helper functions
├── main.py            # Application entry point
└── requirements.txt   # Project dependencies
```

## Advanced Task: Fine-tuning Implementation

The project includes a fine-tuning implementation that allows the model to learn from the USCIS manual directly. This approach uses QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune the LLama model while maintaining reasonable resource requirements.

Key components of the fine-tuning process:

1. **Data Preparation**: Documents are processed into question-answer pairs
2. **Model Configuration**: Uses 8-bit quantization with LoRA for efficient training
3. **Training Process**: Implements supervised fine-tuning with custom prompts

To explore the fine-tuning implementation, see `src/model/fine_tuning.py`.

## System Requirements

- Python 3.8 or higher
- Operating System:
  - Windows 10/11
  - macOS 10.15 or higher
  - Linux (Ubuntu 20.04 or higher recommended)
- Hardware:
  - CPU: 4+ cores recommended
  - RAM: 16GB minimum
  - GPU: Optional, but recommended for training
    - NVIDIA GPU with CUDA support
    - Apple Silicon (M1/M2/M3) for MPS acceleration
