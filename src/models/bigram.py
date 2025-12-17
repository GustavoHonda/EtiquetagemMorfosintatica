from email.policy import default
from itertools import count
from math import e
from src.models.model import PredictionModel
from src.models.smoothing import laplace_smoothing
from pandas import DataFrame 
from collections import defaultdict

class BigramModel(PredictionModel):
    def __init__(self):
        super().__init__()
        self.unigram = {}
        self.bigram = {}
        self.bigram_suavizado = {}

    def train(self, data: DataFrame):
        """
        Treina o modelo com base nos dados fornecidos.
        
        :param data: DataFrame contendo as palavras e IDs de sentença
        """
        # Conta unigramas
        self.unigram = data['form'].value_counts().to_dict()


        # Conta bigramas usando dicionários encadeados
        last_word = None
        last_id = None

        for _, element in data.iterrows():
            current_word = element['form']
            current_id = element['sent_id']

            if last_word is not None and last_id == current_id:
                # Se já existirem bigramas para a palavra anterior, atualiza
                if last_word not in self.bigram:
                    self.bigram[last_word] = {}
                if current_word not in self.bigram[last_word]:
                    self.bigram[last_word][current_word] = 0
                self.bigram[last_word][current_word] += 1

            # Atualiza o "último" word e o ID da sentença
            last_word = current_word
            last_id = current_id

        # Aplica a suavização de Laplace
        self.bigram_suavizado = laplace_smoothing(self.bigram, self.unigram, len(self.unigram))


    def unknown_handling(self, data: DataFrame):
        last_word = None
        last_id = None
        for _, element in data.iterrows():
            if last_word == "<unk>":
                if last_word not in self.bigram_suavizado:
                    self.bigram_suavizado[last_word] = {}
                if element['form'] not in self.bigram_suavizado[last_word]:
                    self.bigram_suavizado[last_word][element['form']] = 0
                self.bigram_suavizado[last_word][element['form']] += 1
            last_word = element['form']
            last_id = element['sent_id']

    def predict(self, test_data: DataFrame):
        # Para palavras unknown percorrer o DataFrame e adicionar nos bigramas suavizados

        self.unknown_handling(test_data)

        results = defaultdict(list)
        last_word = "<s>"
        last_id = None
        for _, element in test_data.iterrows():
            if last_id is not None and element['sent_id'] == last_id:
                next_words = self.bigram_suavizado[last_word]
                predicted_word = max(next_words, key=next_words.get)
                # results.get(element['sent_id'], [])
                results[element['sent_id']].append(predicted_word)
            else:
                # results.get(element['sent_id'], [])
                next_words = self.bigram_suavizado["<s>"]
                predicted_word = max(next_words, key=next_words.get)
                results[element['sent_id']].append(predicted_word)
            last_word = element['form']
            last_id = element['sent_id']
        

        
        return results


    def evaluate(self, test_data: DataFrame):
        predictions = self.predict(test_data)
        expected_predictions = defaultdict(list)
        metrics=defaultdict(list)
        for _, element in test_data.iterrows():
            expected_predictions[element['sent_id']].append(element['form'])

        
        count_gold = defaultdict(int)
        for predicted, expected in zip(predictions.values(), expected_predictions.values()):
            for p, e in zip(predicted, expected):
                if p == e:
                    count_gold[e] += 1

            accuracy = sum(p == e for p, e in zip(predicted, expected)) / len(expected)
            metrics["acuracy"].append(accuracy)
            precision = sum(p in expected for p in predicted) / len(predicted)
            metrics["precision"].append(precision)
            recall = sum(e in predicted for e in expected) / len(expected)
            metrics["recall"].append(recall)
            f1_score = 2 * (precision * recall / (precision + recall))
            metrics["f1_score"].append(f1_score)

        return metrics, count_gold

    def save(self, filepath):
        # Implement save logic for bigram model
        pass

    def load(self, filepath):
        # Implement load logic for bigram model
        pass