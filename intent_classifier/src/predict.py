# src/predict.py
import joblib
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text, lemmatize_text
from vectorization import vectirizer_with_bert

# Загрузка модели
model = joblib.load("modelsWithBERTLemma/intent_model.pkl")
bert = SentenceTransformer("all-MiniLM-L6-v2")

def predict_intent(text: str) -> str:
    text = lemmatize_text(clean_text(text))
    vec = vectirizer_with_bert([text])
    prediction = model.predict(vec)[0]
    return prediction
