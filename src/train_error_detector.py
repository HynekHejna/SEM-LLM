"""train_error_detector.py
Autor: Hynek Hejna
Datum: 11.5.2025
Součástí semestrálního projektu Využítí velkého jazykového modelu pro detekci a opravu chyb v textu.
TUL FM IT, 2025
Popis: Skript pro trénink modelu pro detekci chyb v textu pomocí token classification.
"""

import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch

MODEL_NAME = "ufal/robeczech-base"
DATA_PATH = "src/datasets/detector_dataset_v1.jsonl"
#DATA_PATH = "src/datasets/token_classification_dataset.jsonl"
LABEL_LIST = ["O", "ERR"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)} # mapování labelů na ID
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Načtení tokenizeru a modelu
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Načtení datasetu pro trénink
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

examples = load_data(DATA_PATH)

# Tokenizace a alignace labelů
def tokenize_and_align(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=64)
    labels = []
    word_ids = tokenized.word_ids()
    prev_word_idx = None
    # Z důvodu subword tokenizace je třeba mapovat tokeny na slova pomocí labelů
    for word_idx in word_ids:
        if word_idx is None: # Eliminace tokenů, které nepatří k žádnému slovu
            labels.append(-100)
        elif word_idx != prev_word_idx: #pokud je první token slova, přidáme label
            labels.append(LABEL2ID[example["labels"][word_idx]])
        else:
            labels.append(-100) # Pokud je to další token slova, přidáme -100
        prev_word_idx = word_idx
    tokenized["labels"] = labels
    return tokenized

# Převod na HuggingFace dataset
hf_dataset = Dataset.from_list(examples)
tokenized_dataset = hf_dataset.map(tokenize_and_align)

# Trénovací argumenty
args = TrainingArguments(
    output_dir="./robe-error-detector",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_steps=10,
    logging_steps=5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./robe-error-detector")
