from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch

model_path = "./testing/" 
tokenizer = ElectraTokenizer.from_pretrained(model_path)
model = ElectraForSequenceClassification.from_pretrained(model_path)

def correct_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return "Opravená věta" if predicted_class == 1 else "Ponechat původní větu"

test_sentence = "Mýval šel po dvoře a hledal mýdlový myš."
print(correct_sentence(test_sentence))

