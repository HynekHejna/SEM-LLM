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
    words = sentence.split()
    original_word = words[error_idx]

    # Tokenizace slova na subwordy
    tokenized_word = tokenizer(original_word, add_special_tokens=False)
    num_subtokens = len(tokenized_word.input_ids)

    # Maskování podle počtu subwordů
    masked_words = words.copy()
    masked_words[error_idx] = " ".join(["[MASK]"] * num_subtokens)
    masked_sentence = " ".join(masked_words)

    # Tokenizace celé věty s maskou
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    input_ids = inputs.input_ids

    # Najdi indexy masek
    mask_token_id = tokenizer.mask_token_id
    mask_token_indices = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Top predikce pro každou masku
    top_k = 5
    mask_preds = [
        torch.topk(logits[0, idx], k=top_k).indices.tolist()
        for idx in mask_token_indices
    ]

    # Kombinace predikovaných tokenů
    candidates = list(product(*mask_preds))

    candidate_words = []
    for token_ids in candidates:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")
        candidate_words.append(decoded)

    # Výběr nejlepšího (první)
    best_word = candidate_words[0] if candidate_words else "[NEZNÁMO]"
    corrected_words = words.copy()
    corrected_words[error_idx] = best_word
    corrected_sentence = " ".join(corrected_words)

    return best_word, corrected_sentence