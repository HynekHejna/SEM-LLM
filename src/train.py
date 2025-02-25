from transformers import Trainer, TrainingArguments, ElectraForSequenceClassification, ElectraTokenizer
from datasets import load_dataset

tokenizer = ElectraTokenizer.from_pretrained('Seznam/small-e-czech')
model = ElectraForSequenceClassification.from_pretrained('Seznam/small-e-czech', num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = load_dataset("csv", data_files="czech_grammar_dataset.csv")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.rename_column("correct_label", "labels")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

trainer.train()