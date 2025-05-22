import json
import re
def parse_sentence(sentence):
    """
    Parse a sentence into a list of words.
    """
    return sentence.split()

def parse_text(text):
    """
    Parse a text into a list of sentences.
    """
    text = text.replace("„", "")
    text = text.replace("“", "")
    pattern = r'(?<!\d)\.'
    return re.split(pattern, text)

def parse_to_dataset(text):
    """
    Parse a text into a dataset.
    """
    sentences = parse_text(text)
    dataset = []
    for sentence in sentences:
        words = parse_sentence(sentence)
        label_list = ["O",] * len(words)
        dataset.append({"tokens": words, "labels": label_list})
    return dataset
            
if __name__ == "__main__":
    # with open("src/datasets/base.txt", "r", encoding="utf-8") as file:
    #     text = file.read()
    # dataset = parse_to_dataset(text)
    # with open("src/datasets/detector_dataset_v1.jsonl", "w", encoding="utf-8") as file:
    #     for entry in dataset:
    #         file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open("src/datasets/detector_dataset_v1.jsonl", "r", encoding="utf-8") as file:
        sen_cnt = 0
        tok_cnt = 0
        for line in file:
            entry = json.loads(line)
            tokens = entry["tokens"]
            for token in tokens:
                tok_cnt += 1
            sen_cnt += 1
        print(f"Počet vět: {sen_cnt}, počet tokenů: {tok_cnt}")