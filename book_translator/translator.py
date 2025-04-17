from transformers import MarianMTModel, MarianTokenizer
import torch

def load_model_and_tokenizer(model_name="Helsinki-NLP/opus-mt-ru-en"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_batch(texts, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]