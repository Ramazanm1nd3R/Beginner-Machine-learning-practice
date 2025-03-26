# main.py
from language_pipeline import detect_and_predict

if __name__ == '__main__':
    print('Введите текст (на русском или английском): ')

    while True:
        text = input('> ')
        if text.lower() in ['exit']:
            print('stop')
            break

        result = detect_and_predict(text)
        print(f'sentiment: {result}')