project_root/
├── data/                        # raw/processed data for training/testing
│   ├── raw/                    # исходный датасет (например, hf-загрузка)
│   ├── processed/              # очищенные и размеченные данные
│   └── intents.yaml            # описание интентов в формате YAML
│
├── models/                     # сериализованные модели (pkl, bin и пр)
│   └── intent_model.pkl
│
├── scripts/                    # вспомогательные скрипты
│   ├── download_dataset.py     # загрузка датасета с huggingface
│   └── prepare_data.py         # очистка, нормализация, лемматизация
│
├── src/                        # логика проекта (обработка, обучение, инференс)
│   ├── __init__.py
│   ├── train.py                # обучение модели
│   ├── predict.py              # запуск модели и получение ответа
│   ├── preprocessing.py        # очистка + лемматизация
│   ├── vectorization.py        # TF-IDF или BERT векторизация
│   └── evaluation.py           # метрики (accuracy, f1 и пр)
│
├── tests/                      # юнит-тесты и тестовые сессии (можно сохранять логи/скрины)
│   └── screenshots/
│
├── notebooks/                  # эксперименты, визуализация, исследования
│   └── exploration.ipynb
│
├── main.py                     # CLI/точка входа для запуска предсказания
├── requirements.txt            # зависимости
└── README.md                   # описание проекта