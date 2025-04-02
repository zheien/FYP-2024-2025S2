from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import torch
from datasets import Dataset
import json
import os

def prepare_training_data(data_list, tokenizer):
    """
    data_list should be a list of dictionaries with 'unnormalized' and 'normalized' keys
    Example: [{'unnormalized': 'Text A', 'normalized': 'Text B'}]
    """
    def format_chat(example):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": example['unnormalized']},
            {"role": "assistant", "content": example['normalized']}
        ]
        # Return the formatted text as a string
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    dataset = Dataset.from_list(data_list)
    tokenized_dataset = dataset.map(
        lambda examples: {
            **tokenizer(
                format_chat(examples),
                truncation=True,
                max_length=512,
                padding="max_length"  # Add padding to ensure consistent sequence lengths
            ),
            "labels": [
                (label if label != tokenizer.pad_token_id else -100)
                for label in tokenizer(
                    format_chat(examples),
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )["input_ids"]  # Use input_ids as labels for causal language modeling
            ]
        },
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def main():
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load training data from a JSON file
    training_data_path = os.path.join(os.path.dirname(__file__), "test.json")  # Replace with your file name
    with open(training_data_path, "r") as f:
        training_data = json.load(f)

    # Prepare dataset
    train_dataset = prepare_training_data(training_data, tokenizer)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./finetuned-qwen",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        fp16=True,  # Enable mixed precision training
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()

    # Save the finetuned model
    model.save_pretrained("./finetuned-qwen")
    tokenizer.save_pretrained("./finetuned-qwen")

if __name__ == "__main__":
    main()