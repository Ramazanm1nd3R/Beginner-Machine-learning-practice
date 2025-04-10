import re
import nltk
from natasha import Doc, NewsEmbedding, NewsMorphTagger, MorphVocab, Segmenter
from nltk.corpus import stopwords

nltk.download('stopwords')

class TextProcessor:
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
    
    def lemmatize(self, text: str) -> str:
        """
        Lemmatize and cleaning text
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        
        return " ".join([token.lemma for token in doc.tokens])
    
    def preprocess(self, text: str) -> str:
        """
        Full pipeline: cleaning → lemmatiz
        """
        cleaned = self.clean_text(text)
        return self.lemmatize(cleaned)