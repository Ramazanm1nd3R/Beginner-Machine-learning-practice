from datasets import load_dataset
dataset = load_dataset("KunalEsM/bank_complaint_intent_classifier")
print(dataset['train'].features)