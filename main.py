import argparse
import logging
import subprocess
import signal
import sys
from pathlib import Path

from src.utils.config import load_config
from src.retrieval.embeddings import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='USCIS Chatbot Runner')
    parser.add_argument(
        '--reinitialize',
        action='store_true',
        help='Regenerate FAISS store from PDF'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    return parser

def check_python_version():
    import sys
    
    min_version = (3, 8)
    current = sys.version_info[:2]
    
    if current < min_version:
        raise RuntimeError(
            f"Python {min_version[0]}.{min_version[1]} or higher required; "
            f"you are using {current[0]}.{current[1]}"
        )

def main():
    parser = setup_args()
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Initialize embeddings
    embedding_manager = EmbeddingManager(config)
    try:
        logger.info("Creating embeddings...")
        store_path = embedding_manager.create_embeddings(force=args.reinitialize)
        if store_path:
            logger.info(f"Created new embeddings at {store_path}")
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        return
    
    # Launch Streamlit app
    streamlit_file = Path(__file__).parent / 'src' / 'interface' / 'streamlit_app.py'
    
    logger.info("Launching Streamlit interface...")
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", str(streamlit_file),
        "--", "--config", args.config
    ])
    
    # Define a signal handler to terminate the subprocess
    def signal_handler(sig, frame):
        logger.info("Terminating Streamlit app...")
        process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    process.wait()

if __name__ == '__main__':
    check_python_version()
    main()
