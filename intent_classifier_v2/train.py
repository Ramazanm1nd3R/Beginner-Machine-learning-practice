import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import torch

from scripts.download_dataset import DatasetLoader
from src.preprocessing_pipeline import TextProcessor
from scripts.model_selection import ModelSelector  

# 1. Определение устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 2. Загрузка датасета
loader = DatasetLoader()
train_df, test_df = loader.to_pandas()

# 3. Обработка текста
processor = TextProcessor()
train_df["text"] = train_df["text"].apply(processor.preprocess)
test_df["text"] = test_df["text"].apply(processor.preprocess)

# 4. Подготовка данных
X_train = train_df["text"].tolist()
y_train = train_df["label"].tolist()
X_test = test_df["text"].tolist()
y_test = test_df["label"].tolist()

# 5. Загрузка модели BERT с поддержкой CUDA
model_loader = ModelSelector()  # автоматически определяет устройство
bert_model = model_loader.get_model()

# 6. Векторизация (GPU ускорение)
X_train_vec = bert_model.encode(X_train, batch_size=32, show_progress_bar=True)
X_test_vec = bert_model.encode(X_test, batch_size=32, show_progress_bar=True)

# 7. Обучение и оценка логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
