import yaml
from pathlib import Path
from typing import Dict, Any
import logging

DEFAULT_CONFIG = {
    'paths': {
        'faiss_store': 'data/faiss_store',
        'model_output': 'models/uscis-llama-qlora'
    },
    'training': {
        'num_epochs': 3,
        'batch_size': 1,
        'learning_rate': 2e-4,
        'gradient_accumulation_steps': 8
    },
    'model': {
        'name': 'meta-llama/Llama-3.2-1B',
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05
    }
}

logger = logging.getLogger(__name__)

def get_platform_specific_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust paths based on operating system."""
    import platform
    
    system = platform.system()
    paths = config.get('paths', {})
    
    if system == "Windows":
        for key, path in paths.items():
            paths[key] = path.replace('/', '\\')
    
    return {**config, 'paths': paths}

def load_config(config_path: str = 'config/default.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file or return defaults."""
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                config = {**DEFAULT_CONFIG, **config}
                return get_platform_specific_paths(config)
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
    
    return get_platform_specific_paths(DEFAULT_CONFIG)
