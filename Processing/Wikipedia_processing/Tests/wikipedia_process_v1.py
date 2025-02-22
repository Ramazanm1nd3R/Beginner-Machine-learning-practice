import wikipediaapi
import re
import nltk
from collections import Counter
from razdel import tokenize
from natasha import MorphVocab, Segmenter, NewsMorphTagger, NewsEmbedding, Doc
from nltk.corpus import stopwords

nltk.download('stopwords')

def get_wikipedia_text(title, lang="ru"):
    user_agent = "NLPProject/1.0 (nanana@email.com)"  # Укажи свое название и email
    wiki = wikipediaapi.Wikipedia(language=lang, user_agent=user_agent)

    page = wiki.page(title)
    return page.text if page.exists() else None

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Удаляем знаки препинания
    return text.lower()

def tokenize_text(text):
    return [token.text for token in tokenize(text)]

def remove_stopwords(tokens):
    stop_words = set(stopwords.words("russian"))
    return [word for word in tokens if word not in stop_words]

def lemmatize(tokens):
    morph_vocab = MorphVocab()
    segmenter = Segmenter()
    emb = NewsEmbedding()  # Добавляем векторное представление
    morph_tagger = NewsMorphTagger(emb)  # Новый морфологический теггер

    doc = Doc(" ".join(tokens))  # Объединяем токены в строку
    doc.segment(segmenter)  # Разбиваем на предложения и токены
    
    # Старый метод doc.parse_morph() заменен на doc.tag_morph(), потому что он больше не поддерживается
    # doc.parse_morph()  # устаревший метод

    doc.tag_morph(morph_tagger)  # аткульный метод морфологического анализа

    for token in doc.tokens:
        token.lemmatize(morph_vocab)  # Лемматизируем каждый токен

    return [token.lemma for token in doc.tokens]  # Возвращаем список лемм

def get_top_words(words, n=10):
    return Counter(words).most_common(n)

article_title = "Машинное обучение"  # Можно менять название статьи
text = get_wikipedia_text(article_title)

if text:
    text_cleaned = clean_text(text)
    tokens = tokenize_text(text_cleaned)
    tokens_filtered = remove_stopwords(tokens)
    lemmas = lemmatize(tokens_filtered)
    top_words = get_top_words(lemmas)

    print(f"Топ-10 слов из статьи «{article_title}»:")
    for word, freq in top_words:
        print(f"{word}: {freq}")
else:
    print("Статья не найдена!")
