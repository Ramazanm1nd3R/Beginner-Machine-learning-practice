import pandas as pd
from datasets import load_dataset 
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, dataset_name='KunalEsM/bank_complaint_intent_classifier', split_ratio=0.8):
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        self.raw_dataset = None
        self.train_df = None
        self.test_df = None

    def load(self):
        print(f'Load dataset {self.dataset_name}')
        self.raw_dataset = load_dataset(self.dataset_name)
        return self.raw_dataset
    
    def to_pandas(self):
        if self.raw_dataset is None:
            self.load()

        df = self.raw_dataset['train'].to_pandas()
        self.train_df, self.test_df = train_test_split(df, test_size=1-self.split_ratio, random_state=42)

        return self.train_df, self.test_df