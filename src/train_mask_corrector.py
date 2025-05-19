"""train_mask_corrector.py
Autor: Hynek Hejna
Datum: 11.5.2025
Součástí semestrálního projektu Využítí velkého jazykového modelu pro detekci a opravu chyb v textu.
TUL FM IT, 2025
Popis: Skript pro trénink modelu pro opravu chyb v textu pomocí masked language modeling.
"""
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
import torch

MODEL_NAME = "ufal/robeczech-base"
DATA_PATH = "src/datasets/mask_correction_dataset.jsonl"

# Načtení tokenizeru a modelu
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

# Načtení datasetu pro trénink
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

examples = load_data(DATA_PATH)

# Tokenizace a příprava dat
def tokenize(example):
    prompt = f"Věta: {example['masked_input']} Nápověda: {example['hint']}" # Definice promptu
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=64)

    # Nalezení pozice masky a správného slovo
    mask_index = tokenized.input_ids.index(tokenizer.mask_token_id)
    labels = [-100] * len(tokenized.input_ids)
    target_ids = tokenizer.encode(example['target'], add_special_tokens=False)

    # Vlož správné ID pro masku
    for i, tid in enumerate(target_ids):
        if mask_index + i < len(labels):
            labels[mask_index + i] = tid

    tokenized["labels"] = labels
    return tokenized

# Převod na HuggingFace dataset
hf_dataset = Dataset.from_list(examples)
tokenized_dataset = hf_dataset.map(tokenize)

# Trénovací argumenty
args = TrainingArguments(
    output_dir="./robe-mask-corrector",
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
trainer.save_model("./robe-mask-corrector")
