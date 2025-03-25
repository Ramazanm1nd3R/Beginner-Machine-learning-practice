# train_model.py
from preprocessing import load_and_process_data
from vectorization import vectorize_text, save_vectorizer
from model import train_and_evaluate_model, save_model

if __name__ == "__main__":
    train_df, test_df = load_and_process_data()

    if train_df is None or test_df is None:
        print("error load data")
    else:
        X_train, X_test, y_train, y_test, vectorizer = vectorize_text(train_df, test_df)
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

        save_model(model)
        save_vectorizer(vectorizer)