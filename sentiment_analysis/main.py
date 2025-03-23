# main.py
from preprocessing import load_and_process_data
from vectorization import vectorize_text
from model import train_and_evaluate_model

if __name__ == "__main__":
    # 1. Загрузка и предобработка данных
    train_df, test_df = load_and_process_data()

    if train_df is None or test_df is None:
        print("Ошибка при загрузке данных. Завершение программы.")
    else:
        # 2. Векторизация текста
        X_train, X_test, y_train, y_test = vectorize_text(train_df, test_df)

        # 3. Обучение и оценка модели
        train_and_evaluate_model(X_train, X_test, y_train, y_test)