from hmmlearn import hmm
import numpy as np
import pandas as pd

class HMM:
    def __init__(self, df_dev):
        self.states = list(df_dev['upos'].unique())
        self.num_states = len(self.states)
        self.state_to_index = {s: i for i, s in enumerate(self.states)}
        self.index_to_state = {i: s for i, s in enumerate(self.states)}

        self.observations = list(df_dev['form'].unique())
        self.word_to_index = {w: i for i, w in enumerate(self.observations)}
        self.index_to_word = {i: w for w, i in self.word_to_index.items()}

        self.model = hmm.MultinomialHMM(
            n_components=self.num_states,
            n_iter=50,
            verbose=True
        )

    def train(self, df_train):
        if '<unk>' not in self.word_to_index:
            df_train.loc[len(df_train)] = {
                'form': '<unk>',
                'upos': 'X',
                'sent_id': 'UNK_SENT',
                'text': 'placeholder'
            }
            self.word_to_index['<unk>'] = len(self.word_to_index)
            self.index_to_word[len(self.index_to_word)] = '<unk>'

        X = df_train['form'].apply(lambda w: self.word_to_index.get(w, self.word_to_index['<unk>']))
        X = X.values.reshape(-1, 1)

        states_indices = df_train['upos'].map(self.state_to_index).values

        lengths = df_train.groupby('sent_id').size().tolist()


        self.model.fit(X, lengths=lengths)

        self.emission_matrix = self.model.emissionprob_

    def predict(self, df_eval):
        X = df_eval['form'].apply(lambda w: self.word_to_index.get(w, self.word_to_index['<unk>']))
        X = X.values.reshape(-1, 1)
        logprob, hidden_states = self.model.decode(X, algorithm='viterbi')

        predicted_words = []
        for s in hidden_states:
            emission_probs = self.emission_matrix[s]
            predicted_word_index = np.argmax(emission_probs)
            predicted_words.append(self.index_to_word[predicted_word_index])

        return predicted_words
