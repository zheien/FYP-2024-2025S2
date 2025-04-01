
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

# Specify the path to your locally saved model directory
model_dir = "/home/users/ntu/e220208/hf_models/bert_model/bert_model"  # Path to your model directory

# Load the model and tokenizer from the local directory
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
# Load dataset (use your own dataset or a public one)
dataset = load_dataset("glue", "mrpc")  # Example dataset

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=tokenized_datasets["train"],  # training dataset
    eval_dataset=tokenized_datasets["validation"],  # validation dataset
)

# Train the model
trainer.train()
