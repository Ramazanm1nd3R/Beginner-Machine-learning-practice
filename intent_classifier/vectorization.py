# vectorization.py
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
import os

model_path = 'modelsWithBert/bert_vectorizer.pkl'

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def vectirizer_with_bert(texts):
    return bert_model.encode(texts)

def save_vectorizer():
    joblib.dump(bert_model, model_path)

def load_vectorizer():
    return SentenceTransformer('all-MiniLM-L6-v2')