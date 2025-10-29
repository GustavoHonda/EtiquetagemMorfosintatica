
from abc import ABC, abstractmethod

class PredictionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, data):
        pass
    
    @abstractmethod
    def predict(self, context):
        pass

    @abstractmethod
    def evaluate(self, test_data):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


