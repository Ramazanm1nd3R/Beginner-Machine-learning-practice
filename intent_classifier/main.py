# main.py 
from src.predict import predict_intent

print("введите фразу или выход, для выхода: ")
while True:
    user_input = input("> ")
    if user_input.lower() in ['выход', 'exit']:
        break

    intent = predict_intent(user_input)
    print(f'предсказание намерения {intent}')