# USCIS Chatbot

An intelligent chatbot system designed to answer questions based on the USCIS manual using advanced language models and retrieval-augmented generation.

This was tested on MacOS with an M2 Pro chip, and Windows with an RTX 3060 Ti.

## Features

- Interactive interface using Streamlit
- RAG-based document retrieval using FAISS
- Intelligent context-aware responses using LLama 3.1 8b
- Citation support with page references
- Customizable model temperature and retriever k-value
- Allows for rapid prototyping and versioning through YAML configurations

## Prerequisites

- Python 3.10 (required for compatibility)
- [Ollama](https://ollama.ai/) installed and running
- At least 16GB RAM recommended (32GB used in testing)
- For fine-tuning (optional):
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon (M1/M2/M3) for MPS acceleration

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Jtewen/chatbot-mettle.git
   cd chatbot-mettle
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

4. Download the ollama models:

   On Unix/macOS:
   ```bash
   ollama pull llama3.1:8b && ollama pull nomic-embed-text
   ```

   On Windows:
   ```bash
   ollama pull llama3.1:8b ; ollama pull nomic-embed-text
   ```

5. Download the USCIS manual PDF from:
   `https://www.uscis.gov/book/export/html/68600`
   and place it in the `/data` directory with the name `uscis_manual.pdf`

## Running the Application

1. Ensure your virtual environment is activated

2. Start the application:
   ```bash
   python main.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

## Optional: Fine-tuning Setup (untested)

If you plan to use the fine-tuning functionality:

1. Ensure you are logged into Hugging Face through the Hugging Face CLI (`huggingface-cli login`) with an account that has access to the `meta-llama/Meta-Llama-3.2` model

2. Prepare your training data in the `data` directory (default is `data/uscis_manual.pdf`)

3. Run the fine-tuning script:
   ```bash
   python -m src.model.fine_tuning
   ```

## Technology Stack

### RAG
- **Python**: Core programming language (3.8+)
- **LLama 3.1**: 8B parameter language model for natural language understanding
- **LangChain**: Framework for composing language model applications
- **FAISS**: Vector database for similarity search and document retrieval
- **Streamlit**: Web framework for building the user interface
- **Ollama**: LLM model serving and management platform

### Fine-tuning
- **Python**: Core programming language (3.10+)
- **Meta-Llama 3.2**: 1B parameter language model for natural language understanding
- **QLoRA**: Quantized Low-Rank Adaptation for efficient fine-tuning on limited hardware.
- **Hugging Face**: Model hosting for local download.

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

The project includes a fine-tuning implementation that allows the model to learn directly from the USCIS manual using QLoRA (Quantized Low-Rank Adaptation). This approach significantly reduces memory requirements while maintaining model quality. It also defaults to using Llama 3.2 for a smaller footprint.

I've decided to take a more manual approach to fine-tuning, as opposed to using the Hugging Face Trainer API or Unsloth. This allows for more control over the training process and the ability to implement custom techniques such as multi-format prompt templates and data augmentation. It also reduces the amount of dependencies and external libraries required for the proof of concept. At scale, I would recommend using a more specialized framework such as Unsloth or TRL.

It is untested as I do not have the resources to run it in a timely manner.

### Data Processing Pipeline
- Multi-format prompt templates
- Intelligent text chunking with boundary detection
- Data augmentation with controlled noise injection (typos, formatting variations)
- Dynamic dataset filtering and validation (not implemented)

### Training Architecture
- 4-bit quantization with double quantization
- Advanced LoRA configuration targeting multiple attention layers
- Cosine learning rate scheduling with warm restarts
- Gradient checkpointing and mixed precision training

### Evaluation Framework (not implemented as it would require a tuned working model to properly implement)
- Perplexity and response accuracy metrics
- Citation verification system
- Context relevance scoring
- Automated evaluation pipeline

### Quality Assurance
- Experiment tracking with W&B or tensorboard integration
- Logging and monitoring
- Model checkpoint management
- Performance regression testing

To explore the fine-tuning implementation, see `src/model/fine_tuning.py` and `src/model/text_augmentation.py`.

## System Requirements

- Python 3.10 preferred
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

## Notes and Comments

- The fine-tuning implementation is not yet tested.
- I went overkill with the per-device settings as I was unsure of the hardware this would be tested on.
- The RAG implementation has no caching or memory, as I didn't want to host a database. Memory seemed superfluous for the scope of this project.
- The project structure is a bit overcomplicated for a proof of concept, but it allows for easy expansion and modification.