# main.py
from model import predict_sentiment

if __name__ == "__main__":
    text = input("enter text: ")
    result = predict_sentiment(text)
    print(f"sentiment {result}")