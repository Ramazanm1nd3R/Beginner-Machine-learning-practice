import wikipediaapi
import re
import nltk
from collections import Counter
from razdel import tokenize, sentenize
from natasha import MorphVocab, Segmenter, NewsMorphTagger, NewsEmbedding, Doc, NewsNERTagger
from nltk.corpus import stopwords

# Устанавливаем список стоп-слов
nltk.download('stopwords')


def get_wikipedia_text(title, lang = "ru"):
    """
    Получаем текст со статьей по апи википедии
    """

    user_agent = "NLPProject/1.0 (nanana@email.com)"
    wiki = wikipediaapi.Wikipedia(language=lang, user_agent=user_agent)
    page = wiki.page(title)

    return page.text if page.exists() else None


def clean_text(text):
    """
    Очищаем текст от лишних символов при помощи регулярного выражения
    и приводим к нижнему регистру 
    """

    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()


def tokenize_text(text):
    """
    Разбираем текст на токены
    """

    tokens = tokenize(text)
    tonekize_text = []

    for token in tokens:
        tonekize_text.append(token.text)
    
    return tonekize_text


def sentenize_text(text):
    """
    Разбираем текст на предложения
    """

    sentences = sentenize(text)
    sentenize_text = []

    for sentence in sentences:
        sentenize_text.append(sentence.text)

    return sentenize_text


def remove_stopwords(tokens):
    """
    Удаляет стоп слова из списка токенов 
    """

    stop_words = set(stopwords.words("russian"))
    without_stopwords = []

    for word in tokens:
        if word not in stop_words:
            without_stopwords.append(word)
    
    return without_stopwords


def lemmatize(tokens):
    """
    Лемматизация текста, проводит слова к исходному 
    виду (что бы в перспективе модель затрачевала меньше ресурсов на их распознавание) 
    """

    morph_vocab = MorphVocab()
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    doc = Doc("".join(tokens))
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    lemmas = []
    for token in doc.tokens:
        lemmas.append(token.lemma)
    
    return lemmas


def extract_entities(text):
    """
    Извлекает именованные сущности (NER) из текста.
    """
    segmenter = Segmenter()
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    entities = []

    for span in doc.spans:
        entities.append((span.text, span.type))

    return entities


def get_top_words(words, n = 10):
    """
    Топ n = самых чистых слов
    """

    return Counter(words).most_common(n)



article_title = "Машинное обучение"
text = get_wikipedia_text(article_title)


if text:
    print(f"Текст статьи из Википедии: {article_title} (первые 300 символов)")
    print(text[:300])
    print()

    text_cleaned = clean_text(text)
    print("Очищенный текст (первые 300 символов):")
    print(text_cleaned[:300])
    print()

    sentences = sentenize_text(text_cleaned)
    print(f"Предложения (первые 3): {sentences[:3]}")
    print()

    tokens = tokenize_text(text_cleaned)
    tokens_filtered = remove_stopwords(tokens)
    print(f"Токены без стоп-слов (первые 10): {tokens_filtered[:10]}")
    print()

    lemmas = lemmatize(tokens_filtered)
    print(f"Леммы (первые 10): {lemmas[:10]}")
    print()

    entities = extract_entities(text)
    print("Извлеченные сущности (NER):")
    for entity, entity_type in entities[:10]:
        print(f"{entity} ({entity_type})")
    print()

    top_words = get_top_words(lemmas)
    print(f"Топ-10 слов из статьи «{article_title}»:")
    for word, freq in top_words:
        print(f"{word}: {freq}")
else:
    print("Статья не найдена!")