# vectorization.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

def vectorize_text(train_df, test_df):
    # Используем очищенный текст
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Объединяем тексты из train и test для согласованного векторного пространства
    all_texts = pd.concat([train_df['clean_text'], test_df['clean_text']])
    vectorizer.fit(all_texts)

    X_train = vectorizer.transform(train_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])
    y_train = train_df['label']
    y_test = test_df['label']

    return X_train, X_test, y_train, y_test
