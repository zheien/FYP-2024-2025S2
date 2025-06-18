from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from jiwer import cer,wer
import editdistance
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import Dataset
import json
import os
output_dir = "./finetuned-qwen3"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
def format_chat(example, tokenizer, inference=False):
    messages = []
    try:
        prompt = f"Return the given text verbatim but with punctuations and symbols.\n\n### Text\n{example['normalized']}"
    
    except KeyError:
        print(f"Missing 'normalized' key in example: {example}")
        raise
    if not inference:
        assistant_response = f"### Response\n{example['unnormalized']}"
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    else:
        prompt = f"Return the given text verbatim but with punctuations and symbols.\n\n### Text\n{example['normalized']}"
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=inference, 
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors=None,
    )

    if not inference:
        return {
            "input_ids": input_ids,
            "labels": input_ids.copy()
        }
    else:
        return {
            "input_ids": input_ids
        }

def prepare_training_data(data_list, tokenizer, device):

    dataset = Dataset.from_list(data_list)
    tokenized_dataset = dataset.map(
        lambda example: format_chat(example, tokenizer),
        remove_columns=[],
        batched=False
    )

    print("Data preparation complete.")
    return tokenized_dataset

def prepare_validation_data(data_list, tokenizer, device):
    print("Starting validation data preparation...")
    dataset = Dataset.from_list(data_list)

    tokenized_dataset = dataset.map(
        lambda example: format_chat(example, tokenizer),
        # remove_columns=dataset.column_names,
        remove_columns=[],
        batched=False
    )

    print("Validation data preparation complete.")
    return tokenized_dataset

def main():
    import os
    shard_index = int(os.environ.get("SHARD_INDEX"))  # no default
    num_shards = int(os.environ.get("NUM_SHARDS"))
    file_name = os.environ.get("FILE_NAME")
    print("Loading the model and tokenizer...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model is running on: {device}")

    # Load training data
    training_data_path = os.path.join(os.path.dirname(__file__), "datasets", "spgi_train.json")

    with open(training_data_path, "r") as f:
        training_data = json.load(f)
    train_dataset = load_dataset("json", data_files=training_data, split="train", streaming=True)

    # Load validation data
    validation_data_path = os.path.join(os.path.dirname(__file__), "datasets", "spgi_val.json")
    with open(validation_data_path, "r") as f:
        validation_data  = json.load(f)

    # Load test data
    test_data_path = os.path.join(os.path.dirname(__file__), "datasets", "spgi_test.json")
    with open(test_data_path, "r") as f:
        test_data  = json.load(f)
    
    # Prepare datasets
    train_shard = train_dataset.shard(num_shards=num_shards, index=shard_index)


    train_dataset = prepare_training_data(train_shard, tokenizer, device)



    validation_dataset = prepare_validation_data(validation_data , tokenizer, device)


    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        # max_steps=1000,
        per_device_train_batch_size=5, 
        gradient_accumulation_steps=64,
        learning_rate=1e-5,
        warmup_steps=100,
        logging_steps=1000,
        save_steps=1000,
        save_strategy="no",
        bf16=True,
        evaluation_strategy="epoch",
        eval_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=2,
    )

    # data_collator = DataCollatorForLanguageModeling(
    # tokenizer=tokenizer,
    # mlm=False,
    # pad_to_multiple_of=8,
    # return_tensors="pt",
    # # padding=True
    # )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Training interrupted: {e}")

    print("Saving the finetuned model...")
    model.save_pretrained("./finetuned-qwen3")
    tokenizer.save_pretrained("./finetuned-qwen3")

    print("Evaluating the model on the test dataset...")
    model.eval()
    total_edits = 0
    total_chars = 0

    for i, raw_example in enumerate(test_data): 
        eval_input = format_chat(raw_example, tokenizer, inference=True)
        input_ids = torch.tensor(eval_input['input_ids']).unsqueeze(0).to(device)
        attention_mask = (input_ids != tokenizer.eos_token_id).long()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=50)
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            if "### Response" in decoded:
                generated_text = decoded.split("### Response")[-1].strip()
            else:
                generated_text = decoded.strip()

        prompt_text = raw_example['normalized']
        expected_text = raw_example['unnormalized']

        edits = editdistance.eval(generated_text, expected_text)
        chars = len(expected_text)

        total_edits += edits
        total_chars += chars

        # print(f"Expected: {expected_text}")
        # print(f"Generated: {generated_text}")
        # print(f"Edit Distance: {edits}, Length of GT: {chars}")
        # print("-" * 30)
        with open("eval_spgi.txt", "a", encoding="utf-8") as f:
            f.write(f"Input: {prompt_text}\n")
            f.write(f"Expected: {expected_text}\n")
            f.write(f"Generated: {generated_text}\n")
            f.write(f"Edit Distance: {edits}\n")
            f.write("-" * 60 + "\n")

    global_cer = total_edits / total_chars if total_chars > 0 else 0
    print(f"✅ Global Character Error Rate (EditDistance / TotalChars): {global_cer:.4f}")

if __name__ == "__main__":
    print("Script started...")
    main()



