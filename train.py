# Import necessary libraries
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np
from datetime import datetime

# Define the required constants
NUM_DASHES = 200

# Function to log messages with timestamps
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Function to load and process the dataset
def load_and_process_data():
    # Load OpenAssistant dataset
    dataset = load_dataset("OpenAssistant/oasst1")
    
    log_message("Dataset columns: " + str(dataset["train"].column_names))
    log_message("Sample data: " + str(dataset["train"][0]))
    
    # Process the data into conversation format
    def format_conversation(example):
        # The dataset has 'text' and 'role' columns
        if example['role'] == 'assistant':
            # Get the parent message if it exists
            parent_message = example.get('parent', '')
            return {
                "text": f"Human: {parent_message}\nAssistant: {example['text']}"
            }
        return None
    
    # Filter for assistant responses and map them
    processed_dataset = dataset["train"].filter(lambda x: x['role'] == 'assistant')
    processed_dataset = processed_dataset.map(format_conversation, remove_columns=dataset["train"].column_names)
    return processed_dataset

# Function to tokenize the dataset
def tokenize_dataset(dataset, tokenizer):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256,  # Reduced from 2048, 512
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=4  # Added parallel processing
    )
    return tokenized_dataset

# Function to compute perplexity metrics
def compute_metrics(eval_pred):
    # Remove compute_metrics for now as it's causing issues
    return {}

def main():
    log_message("🚀 Starting the fine-tuning process...")
    
    log_message("📦 Loading configuration and model...")
    # Configure QLoRA with more memory-efficient settings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    log_message("🤖 Loading Phi-2 model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA with correct target modules for phi-2
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Update target modules to match phi-2's architecture
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    log_message("-" * NUM_DASHES)
    log_message("🔄 Processing dataset...")
    # Load and process dataset
    dataset = load_and_process_data()
    log_message(f"📊 Dataset size: {len(dataset)} examples")
    
    log_message("🔤 Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Split dataset into train and validation
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Limit evaluation dataset size
    if len(tokenized_dataset["test"]) > 50:  # Reduced from 100
        tokenized_dataset["test"] = tokenized_dataset["test"].select(range(50))
    
    log_message(f"📈 Train size: {len(tokenized_dataset['train'])} examples")
    log_message(f"🔍 Test size: {len(tokenized_dataset['test'])} examples")

    log_message("-" * NUM_DASHES)
    log_message("⚙️ Configuring training arguments...")
    # Training arguments with reduced memory usage
    training_args = TrainingArguments(
        output_dir="./phi2-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Increased
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,                # Increased
        save_strategy="steps",
        save_steps=1000,                # Increased
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        save_total_limit=2,             # Reduced
        optim="adamw_torch_fused",
        # Added memory optimizations
        deepspeed=None,
        ddp_find_unused_parameters=False,
        group_by_length=True,           # Added
        length_column_name="length",    # Added
        report_to="none",              # Disable wandb/tensorboard
    )

    # Initialize trainer with processing_class instead of tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=None,  # Remove direct tokenizer reference
            mlm=False,
            processing_class=tokenizer  # Use processing_class instead
        ),
    )

    # Calculate metrics before training
    log_message("-" * NUM_DASHES)
    log_message("Starting training...")
    # Train the model
    trainer.train()

    log_message("✨ Training completed!")
    
    # Calculate loss instead of perplexity
    eval_results = trainer.evaluate()
    log_message(f"📊 Final evaluation loss: {eval_results['eval_loss']}")

if __name__ == "__main__":
    log_message("=" * NUM_DASHES)
    log_message("💫 Initializing fine-tuning script...")
    log_message("=" * NUM_DASHES)
    main()
    log_message("=" * NUM_DASHES)
    log_message("✅ Process completed successfully!") 
    log_message("=" * NUM_DASHES)