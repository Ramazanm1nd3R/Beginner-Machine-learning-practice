import re
import nltk
import spacy
from natasha import Doc, NewsEmbedding, NewsMorphTagger, MorphVocab, Segmenter
from nltk.corpus import stopwords

nltk.download('stopwords')

class RussianTextProcessor:
    def __init__(self):
        """
        Init all components for text processing
        """
        self.segmenter = Segmenter()
        self.embedding = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.embedding)
        self.morph_vocab = MorphVocab()
        self.stop_words = stopwords.words('english')

    def clean_text(self, text: str) -> str:
        """
        Cleaning up text from links, special characters, digits and leads to lower register
        """
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Zа-яА-Я\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text
    
    def russian_lemmatize(self, text: str) -> str:
        """
        Lemmatize and cleaning text
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        
        return " ".join([token.lemma for token in doc.tokens])
    
    def russian_preprocess(self, text: str) -> str:
        """
        Full pipeline: cleaning → lemmatiz
        """
        cleaned = self.clean_text(text)
        return self.lemmatize(cleaned)
    
class EnglishTextProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = stopwords.words('english')

    def clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)  # удаление ссылок
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # только буквы
        text = re.sub(r"\d+", "", text)  # удаление цифр
        text = re.sub(r"\s+", " ", text).strip().lower()  # пробелы и lowercase
        return text

    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words and not token.is_punct]
        return " ".join(tokens)

    def preprocess(self, text: str) -> str:
        cleaned = self.clean_text(text)
        return self.lemmatize(cleaned)