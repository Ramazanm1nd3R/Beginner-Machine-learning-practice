import pandas as pd
from scripts.download_dataset import DatasetLoader

class DataPreparer:
    def __init__(self):
        self.loader = DatasetLoader()
        self.train_df = None
        self.test_df = None

    def extract_short_label(self, label: str) -> str:
        try:
            start = label.index('-') + 1
            end = label.lower().index('in')
            return label[start:end].strip().title()
        except ValueError:
            return "Other"

    def load_and_prepare(self):
        self.train_df, self.test_df = self.loader.to_pandas()

        # Добавление сокращённых меток
        self.train_df["short_label"] = self.train_df["label"].apply(self.extract_short_label)
        self.test_df["short_label"] = self.test_df["label"].apply(self.extract_short_label)

        return self.train_df, self.test_df
