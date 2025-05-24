import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM
import json
from correct_sentence import tokenize_and_return_index, mask_and_correct
from datetime import datetime

time = datetime.now().strftime("%m-%d--%H%M")
# Cesty k modelům
DETECTOR_PATH = "Hahacko03/robe-error-detector"
CORRECTOR_PATH = "Hahacko03/robe-mask-corrector"
DATASET_PATH = "src/datasets/eval_dataset_v1.json"
EVAL_DATA_PATH = "./results/"+time+".json"

POSSITIVE = 74
NEGATIVE = 26

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    # Inicializace modelů a tokenizerů
    det_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
    det_model = AutoModelForTokenClassification.from_pretrained(DETECTOR_PATH)

    corr_tokenizer = AutoTokenizer.from_pretrained(CORRECTOR_PATH)
    corr_model = AutoModelForMaskedLM.from_pretrained(CORRECTOR_PATH)

    data = load_data(DATASET_PATH)
        
    # Inicializace proměnných pro vyhodnocení 
    total = len(data)
    det_TP, det_FP, det_FN, det_TN = 0, 0, 0, 0
    corr_score = 0

    for entry in data:
        sentence = entry["sentence"]
        correct_mask = entry["correct_mask"]
        correct_fill = entry["correct_fill"]
        correct_sentece = entry["correct_sentence"]

        #detekce chyby
        error_idx = tokenize_and_return_index(det_model, det_tokenizer, sentence)

        # print("==========================")
        # print(f"Věta: {sentence}")
        # print(f"Správná chyba: {correct_mask}")
        # print(f"Detekovaná chyba: {sentence.split()[error_idx] if error_idx is not None else None}")

        if error_idx is not None:
            mistake = sentence.split()[error_idx]
            fill, filled_sentence = mask_and_correct(corr_model, corr_tokenizer, sentence, error_idx)
        else:    
            mistake = None
            fill = correct_fill
            filled_sentence = correct_sentece

        #evaluace korektoru
        if fill == correct_fill and filled_sentence == correct_sentece:
            corr_score += 1
        # else:
        #     print("Návrh korektury:", repr(filled_sentence))
        #     print("Správná korektura:", repr(correct_sentece))


        #evaluace detektoru
        if mistake is None and correct_mask is None:
            det_TN += 1
        elif mistake == correct_mask:
            det_TP += 1
        elif mistake is None and correct_mask is not None:
            det_FN += 1
        elif mistake != correct_mask:
            det_FP += 1
        
    # Výpočet metrik
    det_precision = det_TP / (det_TP + det_FP) if (det_TP + det_FP) > 0 else 0
    det_recall = det_TP / POSSITIVE
    det_accuracy = (det_TP + det_TN) / total

    print("==========================")
    print(f"Detector - Precision: {det_precision:.2f}, Recall: {det_recall:.2f}, Accuracy: {det_accuracy:.2f}")
    print(f"Corrector - Score: {corr_score / total:.2f}")
    
    # Uložení výsledků
    results = {
        "training_data_detector": "doplnit...",
        "training_data_corrector": "doplnit...",
        "dataset": DATASET_PATH,
        "number_of_sentences": total,
        "number_of_errors": POSSITIVE,
        "detector": {
            "precision": round(det_precision,4),
            "recall": round(det_recall,4),
            "accuracy": round(det_accuracy,4)
        },
        "corrector": {
            "score": round(corr_score / total,4)
        }
    }

    save_results(results, EVAL_DATA_PATH)
