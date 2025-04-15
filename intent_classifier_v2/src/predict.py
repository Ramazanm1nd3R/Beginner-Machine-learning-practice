import joblib
import os

from src.preprocessing_pipeline import EnglishTextProcessor
from scripts.model_selection import ModelSelector

class IntentPredictor:
    def __init__(self, model_path: str = 'models/intent_modelN3_with_spacy_lemma.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'File not found {model_path}')
        
        self.model = joblib.load(model_path)
        self.processor = EnglishTextProcessor()
        selector = ModelSelector()
        self.vectorizer = selector.get_model()
    
    def predict(self, text: str) -> str:
        clean_text = self.processor.preprocess(text)
        vector = self.vectorizer.encode([clean_text])
        prediction = self.model.predict(vector)
        return prediction[0]