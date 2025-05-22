from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Načtení modelu a tokenizeru
tokenizer = AutoTokenizer.from_pretrained("robe-error-detector", add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained("robe-error-detector")

# Testovací věty
test_sentences = [
    "kdo podle vás nese viku za neštěstí ve studénce",
    "to co jsme zvsřejnili byl jen první odhad",
    "přemýšleli jste už o odškodnění pro pozůstalé a zraněné"
]

def detect_errors_in_sentence(sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True) #tokenzace věty, uložení mapování tokenů (víme které tokeny patří k jakým slovům)
    offset_mapping = inputs.pop("offset_mapping") #odtranění mapování tokenů z inputu
    outputs = model(**inputs).logits #spustění modelu na tokenizovaných datech, predikce pro každý token (0 - O, 1 - ERR)
    predictions = torch.argmax(outputs, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) #převede input_ids na textové tokeny
    offsets = offset_mapping[0].tolist() #získání offsetů pro větu

    print(f"\nVěta: {sentence}")
    for idx, (token, pred, (start, end)) in enumerate(zip(tokens, predictions, offsets)):
        if token in tokenizer.all_special_tokens: #přeskočení speciálních tokenů
            continue
        word = sentence[start:end] #získání slova z věty pomocí offsetů
        if pred != 0: #pokud je predikce ERR
            print(f"  - {word} (chyba)")

# Spuštění testování
for sentence in test_sentences:
    detect_errors_in_sentence(sentence)
