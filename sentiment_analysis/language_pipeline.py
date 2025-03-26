# language_pipeline.py
from langdetect import detect
from preprocessing import clean_text, lemmatize_text
from vectorization import load_vectorizer
from model import load_model
from transformers import MarianMTModel, MarianTokenizer

# Инициализация модели перевода
model_name = 'Helsinki-NLP/opus-mt-en-ru'
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator = MarianMTModel.from_pretrained(model_name)

# Функция перевода
def translate_en_to_ru(text):
    tokens = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    translation = translator.generate(**tokens)
    return tokenizer.decode(translation[0], skip_special_tokens=True)

# Общая обработка текста и предсказание
def detect_and_predict(text):
    lang = detect(text)

    if lang == 'en':
        text = translate_en_to_ru(text)

    clean = clean_text(text)
    lemmatized = lemmatize_text(clean)

    vectorizer = load_vectorizer()
    model = load_model()

    X = vectorizer.transform([lemmatized])
    pred = model.predict(X)[0]

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map.get(pred, "Unknown")


# -- В РАЗРАБОТКЕы --