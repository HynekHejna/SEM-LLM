import json
import random
import string

MISTAKE_PROB = 0.75
MIN_LENGHT = 3
MAX_LENGHT = 10
NUMBER_OF_SENTENCES = 20000

def encode_text(file_path):
    """
    Načte text v windows-1250 a převede ho na utf-8.
    """
    with open(file_path, "r", encoding="windows-1250") as file:
        text = file.read()
    with open("src/datasets/backup/train_text_korpus_big_utf8.txt", "w", encoding="utf-8") as file:
        file.write(text)
    return

def cherrypick(file_path):
    base_text = [] 

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.readlines()
        filtered_lines = [line for line in text if MIN_LENGHT < len(line.split()) < MAX_LENGHT]
        base_text = filtered_lines[:NUMBER_OF_SENTENCES]

    with open("src/datasets/backup/train_cherrypicked.txt", "w", encoding="utf-8") as file:
        for i in range(0, int(NUMBER_OF_SENTENCES * 0.9)):
            line = base_text[i]
            file.write(line)
    with open("src/datasets/backup/eval_cherrypicked.txt", "w", encoding="utf-8") as file:
        for i in range(int(NUMBER_OF_SENTENCES * 0.9), NUMBER_OF_SENTENCES):
            line = base_text[i]
            file.write(line)
    return

def introduce_typo(word):
    if len(word) == 0:
       return word
    idx = random.randint(0, len(word) - 1)
    if word[idx].isalpha():
        typo_letter = random.choice([c for c in (string.ascii_lowercase + "áčďéěíňóřšťúůýž") if c != word[idx].lower()])
        return word[:idx] + typo_letter + word[idx+1:]
    return word

def create_mask_dataset(tokens, correct_word, wrong_word,idx):
        tokens[idx] = "[MASK]"
        sentence = " ".join(tokens)
        data = {
            "masked_input": sentence,
            "hint": wrong_word,
            "target": correct_word
        }
        return data

def make_datasets(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.readlines()
        text = [line.strip() for line in text]

    masked = []
    output = []

    for line in text:
        tokens = line.split()
        labels = ["O"] * len(tokens)
        idx = random.randint(0, len(tokens) - 1)
        word = ""
        correct_word = tokens[idx]
        if tokens and random.random() < MISTAKE_PROB:    
            labels[idx] = "ERR"
            tokens[idx] = introduce_typo(tokens[idx])
            word = tokens[idx]
        else:
            word = introduce_typo(tokens[idx])
        masked.append(create_mask_dataset(tokens.copy(), correct_word, word, idx))
        output.append({"tokens": tokens, "labels": labels})
        

    with open("src/datasets/detector_dataset_v3.jsonl", "w", encoding="utf-8") as fout:
        fout.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in output)
    with open("src/datasets/corrector_dataset_v3.jsonl", "w", encoding="utf-8") as fout:
        fout.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in masked)

def make_eval_dataset(file_path):
    data = []
    possitive = 0
    negative = 0
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.readlines()
        for sentece in text:
            sentece = sentece.rstrip("\n")  
            tokens = sentece.split()
            idx = random.randint(0, len(tokens) - 1)
            if tokens and random.random() < MISTAKE_PROB:
                correct_fill = tokens[idx]
                correct_mask = introduce_typo(tokens[idx]) 
                tokens[idx] = correct_mask
                possitive +=1
            else:
                negative +=1
                correct_fill = "None"
                correct_mask = "None"
            correct_sentence =  sentece
            sentence = " ".join(tokens)
            data.append({
                "sentence": sentence,
                "correct_mask": correct_mask,
                "correct_fill": correct_fill,
                "correct_sentence": correct_sentence
            })
    with open("src/datasets/eval_dataset_v3.json", "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
    print(f"Positive: {possitive}, Negative: {negative}")

if __name__ == "__main__":
    #encode_text("src/datasets/backup/train_text_korpus_big.txt")
    cherrypick("src/datasets/backup/train_text_korpus_big_utf8.txt")
    make_datasets("src/datasets/backup/train_cherrypicked.txt")
    make_eval_dataset("src/datasets/backup/eval_cherrypicked.txt")