import numpy as np
from numpy import ones
from collections import Counter
from pandas import DataFrame

class MarkovModel:
    def __init__(self, vocab_size, num_states=17):
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.word_to_index = {}  
        self.index_to_word = {}  

        self.A = ones((num_states, num_states))  
        self.B = ones((num_states, vocab_size))  
        self.pi = ones(num_states)  

        self.normalize_matrices()

    def normalize_matrices(self):
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.B /= self.B.sum(axis=1, keepdims=True)
        self.pi /= self.pi.sum()

    def word_to_index_mapping(self, sample):
        for word in sample:
            if word not in self.word_to_index:
                index = len(self.word_to_index)
                self.word_to_index[word] = index
                self.index_to_word[index] = word

    def forward(self, sample):
        size = len(sample)
        alpha = np.zeros((size, self.num_states))
        # Inicializa alpha[0] 
        alpha[0] = self.pi * self.B[:, sample[0]]
        for i in range(1, size):
            for j in range(self.num_states):
                alpha[i, j] = np.sum(alpha[i-1] * self.A[:, j]) * self.B[j, sample[i]]
        return alpha

    def backward(self, sample):
        size = len(sample)
        beta = np.ones((size, self.num_states))
        for i in range(size-2, -1, -1):
            for j in range(self.num_states):
                beta[i, j] = np.sum(self.A[j, :] * self.B[:, sample[i+1]] * beta[i+1])
        return beta

    def compute_gamma(self, alpha, beta):
        size = len(alpha)
        gamma = np.zeros((size, self.num_states))
        for i in range(size):
            norm_factor = np.sum(alpha[i] * beta[i])
            gamma[i] = (alpha[i] * beta[i]) / norm_factor
        return gamma

    def compute_xi(self, alpha, beta, sample):
        size = len(sample)
        xi = np.zeros((size-1, self.num_states, self.num_states))

        for i in range(size-1):
            norm_factor = np.sum(alpha[i] * self.A * self.B[:, sample[i+1]] * beta[i+1])
            for j in range(self.num_states):
                for k in range(self.num_states):
                    xi[i, j, k] = (alpha[i, j] * self.A[j, k] * self.B[k, sample[i+1]] * beta[i+1, k]) / norm_factor

        return xi

    def update_parameters(self, gamma, xi, sample):
        size = len(sample)

        # Atualiza a distribuição inicial π
        self.pi = gamma[0]

        # Atualiza a matriz de transição A
        for i in range(self.num_states):
            for j in range(self.num_states):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                self.A[i, j] = numerator / denominator

        # Atualiza a matriz de emissão B
        for j in range(self.num_states):
            for k in range(self.vocab_size):
                numerator = np.sum(gamma[sample == k, j])
                denominator = np.sum(gamma[:, j])
                self.B[j, k] = numerator / denominator

    def train(self, data: DataFrame, num_iterations=10):
        """Algoritmo Baum-Welch"""
        for iteration in range(num_iterations):
            print(f"Iteration {iteration+1}/{num_iterations}")

            total_gamma = np.zeros((len(data), self.num_states))  # Inicializa com o tamanho correto
            total_xi = np.zeros((len(data), self.num_states, self.num_states))

            for _, sentence in data.groupby('sent_id'):
                sample = np.array([word for word in sentence['form']])  # Extrai as palavras da sentença
                self.word_to_index_mapping(sample)  # Cria o mapeamento de palavras para índices

                # Converte as palavras para seus índices numéricos
                sample_indices = [self.word_to_index[word] for word in sample]

                print(f"sample: {sample}")
                print(f"sample indices: {sample_indices}")

                # Calcula alpha, beta, gamma e xi
                alpha = self.forward(sample_indices)
                beta = self.backward(sample_indices)
                gamma = self.compute_gamma(alpha, beta)
                xi = self.compute_xi(alpha, beta, sample_indices)

                # Verifique se a forma de gamma é compatível antes de somar
                print(f"gamma shape: {gamma.shape}")
                print(f"total_gamma shape: {total_gamma.shape}")

                if gamma.shape[0] == total_gamma.shape[0]:  # Verifique se os tamanhos são compatíveis
                    total_gamma += gamma
                else:
                    print(f"Warning: Incompatible shapes. Skipping summation for this sentence.")
                
                total_xi += xi

            self.update_parameters(total_gamma, total_xi, data)
            self.normalize_matrices()


    def predict(self, sample):
        """Algoritmo Viterbi"""
        size = len(sample)
        viterbi = np.zeros((size, self.num_states))
        backpointer = np.zeros((size, self.num_states), dtype=int)

        viterbi[0] = self.pi * self.B[:, sample[0]]

        for i in range(1, size):
            for j in range(self.num_states):
                prob = viterbi[i-1] * self.A[:, j] * self.B[j, sample[i]]
                viterbi[i, j] = np.max(prob)
                backpointer[i, j] = np.argmax(prob)

        best_path = np.zeros(size, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])

        for i in range(size-2, -1, -1):
            best_path[i] = backpointer[i+1, best_path[i+1]]

        return best_path   


    def evaluate(self, test_data: DataFrame, predicted_labels):
        
        precision = {}
        recall = {}
        f_score = {}
        accuracy = 0
        label_counts = Counter()
        correct_labels = Counter()
        word_errors = Counter()
        total_words = 0
        correct_words = 0

        for sentence, predicted in zip(test_data, predicted_labels):
            for true_word, true_label, predicted_label in zip(sentence['forms'], sentence['gold_labels'], predicted):
                total_words += 1
                if true_label == predicted_label:
                    correct_words += 1
                    correct_labels[true_label] += 1
                else:
                    word_errors[true_word] += 1
                label_counts[true_label] += 1

        # Micro
        for label in label_counts:
            true_pos = correct_labels[label]
            false_pos = label_counts[label] - true_pos
            false_neg = sum(1 for w, gold in zip(test_data, predicted_labels) if gold != label)
            precision[label] = false_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall[label] = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0
        
        # Macro
        avg_precision = np.mean(list(precision.values()))
        avg_recall = np.mean(list(recall.values()))
        avg_f_score = np.mean(list(f_score.values()))
        accuracy = correct_words / total_words if total_words > 0 else 0
        hardest_words = word_errors.most_common(10)

        return {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f_score": avg_f_score,
            "accuracy": accuracy,
            "hardest_words": hardest_words
        }

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass
