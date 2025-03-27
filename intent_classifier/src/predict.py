# src/predict.py
import joblib
from sentence_transformers import SentenceTransformer

model = joblib.load("modelsWithBert/intent_model.pkl")
bert = SentenceTransformer("all-MiniLM-L6-v2")

def predict_intent(text: str) -> str:
    vec = bert.encode([text])
    prediction = model.predict(vec)[0]
    return prediction
