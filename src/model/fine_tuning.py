from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from trl import SFTTrainer
from pathlib import Path
from src.utils.config import load_config
import random
from src.model.text_augmentation import introduce_typos, add_formatting_variations

def prepare_training_data(config_path: str = "config/default.yaml") -> Dataset:
    """Prepare training data with advanced processing techniques."""
    config = load_config(config_path)
    pdf_path = Path(config['paths']['pdf_path'])
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Training data not found at {pdf_path}")
    
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    
    # Advanced text splitting with intelligent boundary detection
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    texts = splitter.split_documents(documents)
    
    # Enhanced prompt engineering with multiple formats
    training_data = []
    for doc in texts:
        # Direct QA format
        training_data.append({
            'text': f'Question: What can you tell me about {doc.page_content[:50]}?\nAnswer: {doc.page_content}'
        })
        
        # Instruction format
        training_data.append({
            'text': f'Instruction: Explain the following USCIS policy:\n{doc.page_content}\nResponse:'
        })
        
        # Citation format
        training_data.append({
            'text': f'Context: Page {doc.metadata["page"]}\nQuestion: What does this section state?\nAnswer: {doc.page_content}'
        })
    
    dataset = Dataset.from_list(training_data)
    
    # Add data augmentation and filtering
    dataset = dataset.map(augment_data)
    dataset = dataset.filter(lambda x: len(x['text'].split()) > 50)
    
    return dataset

def augment_data(example):
    """Apply data augmentation techniques."""
    text = example['text']
    
    # Add random noise (typos, formatting variations)
    if random.random() < 0.1:
        text = introduce_typos(text)
    
    # Add formatting variations
    if random.random() < 0.15:
        text = add_formatting_variations(text)
    
    return {'text': text}

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

def configure_training(model_name: str = "meta-llama/Llama-3.2-1B"):
    """Configure advanced training parameters."""
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    
    # Advanced LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        fan_in_fan_out=False,
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    training_args = TrainingArguments(
        output_dir="./uscis-llama-qlora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.05,
        optim="adamw_torch",
        fp16=True,
        gradient_checkpointing=True,
        group_by_length=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )
    
    return bnb_config, lora_config, training_args

# def evaluate_model(model, eval_dataset):
#     """Implement comprehensive model evaluation."""
#     metrics = {
#         'perplexity': [],
#         'response_accuracy': [],
#         'citation_accuracy': [],
#         'context_relevance': []
#     }
    
#     evaluator = USCISEvaluator(
#         reference_pdf="data/uscis_manual.pdf",
#         metrics=metrics
#     )
    
#     results = evaluator.evaluate(model, eval_dataset)
    
#     # Log results to wandb or tensorboard
#     log_metrics(results)
    
#     return results

if __name__ == "__main__":
    setup_environment()
    train_model()
