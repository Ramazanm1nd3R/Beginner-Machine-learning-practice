# preprocessing.py
import pandas as pd
import os
import re

# --- Пути к файлам ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


# --- Очистка текста ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # удаляем ссылки
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", "", text)  # оставляем буквы/цифры
    text = re.sub(r"\s+", " ", text)  # убираем лишние пробелы
    return text.strip().lower()


# --- Подготовка датафрейма ---
def prepare_dataframe(df):
    df = df[["text", "sentiment"]].dropna()

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["sentiment"].map(label_map)

    df["clean_text"] = df["text"].apply(clean_text)

    return df


# --- Основной алгоритм ---
def load_and_process_data():
    try:
        train_raw = pd.read_csv(TRAIN_PATH, encoding="cp1252")
        test_raw = pd.read_csv(TEST_PATH, encoding="cp1252")
    except Exception as e:
        print(f"Ошибка при загрузке CSV файлов: {e}")
        return None, None

    train_df = prepare_dataframe(train_raw)
    test_df = prepare_dataframe(test_raw)

    return train_df, test_df
