# src/predict.py
import joblib
import os

MODEL_PATH = os.path.join('models', 'intent_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_intent(text: str) -> str:
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return prediction