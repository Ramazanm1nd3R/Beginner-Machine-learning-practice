# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from vectorization import vectirizer_with_bert
from preprocessing import clean_text, lemmatize_text
import joblib
import os

df = pd.read_csv('data/intent.csv')

df = df.dropna()
texts = df['text']
labels = df['intent']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


# Очистка и лемматизация 
X_train = X_train.apply(clean_text).apply(lemmatize_text)
X_test = X_test.apply(clean_text).apply(lemmatize_text)

# Векторизация текстаtfidf_vectorizer.pkl
# vectorizer = TfidfVectorizer()
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# векторизация с BERT
X_train_vec = vectirizer_with_bert(X_train.tolist())
X_test_vec = vectirizer_with_bert(X_test.tolist())

# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Оценка качества
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Сохранение модели и векторайзера
os.makedirs("modelsWithBERTLemma", exist_ok=True)
joblib.dump(model, "modelsWithBERTLemma/intent_model.pkl")
