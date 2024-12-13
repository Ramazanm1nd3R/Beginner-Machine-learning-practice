import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from model import predict_image  # Импорт функции предсказания из model.py

# Интерфейс рисования
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        # Canvas для рисования
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        # Кнопки
        self.button_predict = tk.Button(root, text="Recognize", command=self.recognize)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        # Метка для отображения результата
        self.label_result = tk.Label(root, text="Draw a digit and press 'Recognize'")
        self.label_result.pack()

        # Изображение и инструмент рисования
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Привязка событий рисования
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = 6  # Размер кисти
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def recognize(self):
        # Преобразование изображения в градации серого
        gray_image = ImageOps.grayscale(self.image)
        result = predict_image(gray_image)  # Вызов функции из model.py
        self.label_result.config(text=f"Recognized digit: {result}")

    def clear_canvas(self):
        # Очистка холста и изображения
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a digit and press 'Recognize'")

# Запуск интерфейса
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()