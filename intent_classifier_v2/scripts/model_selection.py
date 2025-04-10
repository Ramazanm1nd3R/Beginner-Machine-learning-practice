from sentence_transformers import SentenceTransformer
import torch

class ModelSelector:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация модели по имени и устройству (CPU/GPU)
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model()

    def _load_model(self):
        """
        Загрузка модели с учётом устройства
        """
        print(f'Загрузка модели {self.model_name} на {self.device}')
        return SentenceTransformer(self.model_name, device=self.device)

    def get_model(self):
        """
        Вернуть инициализированную модель
        """
        return self.model
