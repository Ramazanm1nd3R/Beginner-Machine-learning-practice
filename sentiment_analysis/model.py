# model.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import clean_text
from vectorization import load_vectorizer
import joblib
import os

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model):
    joblib.dump(model, "models/sentiment_model.pkl")

def load_model():
    return joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Оценка модели ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

    return model

def predict_sentiment(text: str) -> str:
    model = load_model()
    vectorizer = load_vectorizer()

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]

    label_map = {
        0: "negative", 
        1: 'neutral', 
        2: 'positive'
        }
    
    return label_map[pred]

    