import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Шаг 2: Определение архитектуры нейронной сети
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        """
        Архитектура:
        - Слой 1: Сверточный слой (conv1) с 8 фильтрами 3x3, шаг 1, padding 1.
        - Слой 2: MaxPooling с ядром 2x2 и шагом 2.
        - Слой 3: Сверточный слой (conv2) с 16 фильтрами 3x3, шаг 1, padding 1.
        - Слой 4: MaxPooling с ядром 2x2 и шагом 2.
        - Полносвязный слой: Выход 10 классов.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        """
        Forward-проход: обработка входных данных через слои.
        """
        x = F.relu(self.conv1(x))  # Первый сверточный слой + активация ReLU
        x = self.pool(x)           # MaxPooling
        x = F.relu(self.conv2(x))  # Второй сверточный слой + активация ReLU
        x = self.pool(x)           # MaxPooling
        x = x.view(x.size(0), -1)  # Преобразование вектора в одномерный (flatten)
        x = self.fc1(x)            # Полносвязный слой
        return x

# Шаг 5: Функция обучения
def train_model():
    # Устройство для вычислений (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Гиперпараметры
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10

    # Загрузка данных MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели, функции потерь и оптимизатора
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Цикл обучения
    for epoch in range(num_epochs):
        print(f"Эпоха [{epoch + 1}/{num_epochs}]")
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data, targets = data.to(device), targets.to(device)

            # Forward-проход
            scores = model(data)
            loss = criterion(scores, targets)

            # Обратное распространение и обновление весов
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Сохранение обученной модели
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Модель сохранена как 'mnist_cnn.pth'")
    return model

# Функция предсказания для одного изображения
def predict_image(image):
    """
    Принимает изображение (PIL Image), обрабатывает его и возвращает предсказанный класс.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(image)
        _, prediction = scores.max(1)
    return prediction.item()

# Шаг 9: Проверка точности модели
def check_accuracy(loader, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()  # Переключение модели в режим оценки

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = float(num_correct) / float(num_samples) * 100
    print(f"Точность: {accuracy:.2f}%")
    model.train()  # Возвращение в режим обучения

if __name__ == "__main__":
    # Обучение модели
    model = train_model()

    # Тестирование модели
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    check_accuracy(test_loader, model)