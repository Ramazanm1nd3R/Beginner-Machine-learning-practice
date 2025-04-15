from src.predict import IntentPredictor

if __name__ == "__main__":
    while True:
        predictor = IntentPredictor()
        print("Введите текст обращения:")
        
        user_input = input(">>> ")
        if user_input.lower() in ['stop', 'exit']:
            break
        
        result = predictor.predict(user_input)
        print(f"🧠 Предсказанный intent: {result}")