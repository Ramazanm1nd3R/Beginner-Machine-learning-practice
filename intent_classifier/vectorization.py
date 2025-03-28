# vectorization.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def vectirizer_with_bert(texts):
    return model.encode(texts)