# preprocessing.py

import re
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)                  # удаление ссылок
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", "", text)     # удаление спецсимволов
    text = re.sub(r"\s+", " ", text)                     # лишние пробелы
    return text.strip().lower()

def lemmatize_text(text: str) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    return " ".join([token.lemma for token in doc.tokens])