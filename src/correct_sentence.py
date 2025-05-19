import torch
from itertools import product

def tokenize_and_return_index(model, tokenizer, sentence):
    tokens = sentence.split()
    encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
    outputs = model(**encoding)
    predictions = torch.argmax(outputs.logits, dim=2)[0]  

    error_idx = None
    for idx, pred in enumerate(predictions):
        if pred.item() == 1 and encoding.word_ids()[idx] is not None:
            error_idx = encoding.word_ids()[idx]
            break
    return error_idx

def mask_and_correct(model, tokenizer, sentence, error_idx):
    #příprava věty pro korekturu
    original_word = sentence.split()[error_idx]
    masked_tokens = sentence.split().copy()
    tokenized = tokenizer(original_word, add_special_tokens=False)
    num_subtokens = len(tokenized["input_ids"])
    masked_tokens[error_idx] = " ".join(["[MASK]"] * num_subtokens)
    masked_sentence = " ".join(masked_tokens)

    inputs = tokenizer(masked_sentence, return_tensors="pt")
    mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    top_k = 5
    mask_preds = []
    for idx in mask_token_indices:
        top_tokens = torch.topk(logits[0, idx], k=top_k).indices.tolist()
        mask_preds.append(top_tokens)

    # Vytvoření kombinací predikovaných tokenů
    candidates = list(product(*mask_preds))
    
    candidate_words = []
    for combo in candidates:
        decoded = tokenizer.decode(combo, skip_special_tokens=True).strip()
        candidate_words.append(decoded)

    print("Návrhy oprav:")
    for word in candidate_words[:5]:  # omezíme výpis
        print(f" - {word}")

    # První kandidát jako oprava
    best_word = candidate_words[0] if candidate_words else "[NEZNÁMO]"
    corrected_tokens = sentence.split()
    corrected_tokens[error_idx] = best_word
    corrected_sentence = " ".join(corrected_tokens)



    return best_word, corrected_sentence