from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from trl import SFTTrainer
from pathlib import Path
from src.utils.config import load_config

def prepare_training_data(config_path: str = "config/default.yaml") -> Dataset:
    """Prepare training data from the USCIS manual."""
    config = load_config(config_path)
    pdf_path = Path(config['paths']['pdf_path'])
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Training data not found at {pdf_path}")
    
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    
    texts = splitter.split_documents(documents)
    
    # Format into question-answer pairs for training
    training_data = [{
        'text': f'Question: What can you tell me about {doc.page_content[:50]}?\nAnswer: {doc.page_content}'
    } for doc in texts]
    
    return Dataset.from_list(training_data)

def train_model():
    """Fine-tune LLama model on USCIS manual content using QLoRA."""
    device = get_device()
    print(f"Using device: {device}")
    
    train_dataset = prepare_training_data()

    model_kwargs = {
        "device_map": {"": device},
        "trust_remote_code": True
    }

    # Configure device-specific settings
    if device == "cuda":
        # CUDA-specific settings remain the same
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        model_kwargs.update({
            "quantization_config": bnb_config,
            "torch_dtype": torch.float16
        })
    else:
        # MPS and CPU settings
        model_kwargs.update({
            "torch_dtype": torch.float32,
            "use_cache": False  # Disable KV cache for training
        })

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        **model_kwargs
    )

    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA with reduced complexity for MPS
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05 if device == "cuda" else 0.0,  # Disable dropout for MPS
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./uscis-llama-qlora",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        optim="adamw_torch",
        gradient_checkpointing=False if device == "mps" else True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=256
    )
    
    trainer.train()
    trainer.save_model("./uscis-llama-qlora-final")

def get_device() -> str:
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def setup_environment():
    """Configure environment variables based on system."""
    import os
    import platform
    
    # Common settings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Mac-specific settings
    if platform.system() == "Darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

if __name__ == "__main__":
    setup_environment()
    train_model()
