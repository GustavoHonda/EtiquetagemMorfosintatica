from email.policy import default
import numpy as np
from numpy import ones
from collections import Counter
from pandas import DataFrame
from collections import defaultdict

class MarkovModel:
    def __init__(self, df):
        
        self.vocab = df['form'].unique().tolist()
        self.states = df['upos'].unique().tolist()
        self.num_states = len(self.states)
        self.vocab_size = len(self.vocab)
        # Matriz A: 17*17*17 = 289 linhas e 17 colunas (linhas somam 1)
        # Matriz B: 17*len(dicionário) = 17 linhas e m colunas (linhas somam 1)
        # MatrizPi: 17 = 1 linha e 17 colunas (linhas somam 1)
        self.init_A(df)
        self.init_B(df)
        self.init_pi(df)

        self.init_matrices()

        self.compound_index = {
            (i, j): i * self.num_states + j
            for i in range(self.num_states)
            for j in range(self.num_states)
        }

        self.state_to_index = {s: i for i, s in enumerate(self.states)}
        self.index_to_state = {i: s for i, s in enumerate(self.states)}



    def init_A(self,df):
        transitions = defaultdict(lambda: defaultdict(int))
        sentence_id = None
        lastlastupos = None
        lastupos = None

        for row in df.itertuples():
            if sentence_id != row.sent_id:
                lastlastupos = None
                lastupos = None
                sentence_id = row.sent_id
            if lastupos is not None and lastlastupos is not None:
                transitions[(lastlastupos, lastupos)][row.upos] += 1
            lastlastupos = lastupos
            lastupos = row.upos

        for (lastlastupos, lastupos), counts in transitions.items():
            total = sum(counts.values())
            for upos in counts:
                counts[upos] /= total

        
        assert abs(sum(transitions[('NOUN', 'VERB')].values()) - 1.0) < 1e-8 or sum(transitions[('NOUN', 'VERB')].values()) == 0.0
        self.A_dict = transitions

    def init_B(self,df):
        emission_counts = defaultdict(lambda: defaultdict(int))

        for row in df.itertuples():
            emission_counts[row.upos][row.form] += 1

        for upos, counts in emission_counts.items():
            total = sum(counts.values())
            for form in counts:
                counts[form] /= total

        self.B_dict = emission_counts

    def init_pi(self,df):
        state_count = defaultdict(int)
        for row in df.itertuples():
            state_count[row.upos] += 1

        pi = {state: count/len(df) for state, count in state_count.items()}
        self.pi_dict = pi

    def init_matrices(self):
        self.A_matrix = np.zeros((self.num_states * self.num_states, self.num_states))
        self.B_matrix = np.zeros((self.num_states, self.vocab_size))
        self.pi_matrix = np.zeros(self.num_states)

        state_to_index = {state: idx for idx, state in enumerate(self.states)}
        word_to_index = {word: idx for idx, word in enumerate(self.vocab)}

        # Fill A matrix
        for (lastlastupos, lastupos), counts in self.A_dict.items():
            for upos, prob in counts.items():
                i = state_to_index[lastupos]
                j = state_to_index[upos]
                self.A_matrix[i, j] = prob

        # Fill B matrix
        for upos, counts in self.B_dict.items():
            for form, prob in counts.items():
                i = state_to_index[upos]
                k = word_to_index[form]
                self.B_matrix[i, k] = prob

        # Fill pi vector
        for upos, prob in self.pi_dict.items():
            i = state_to_index[upos]
            self.pi_matrix[i] = prob

        self.word_to_index = word_to_index
        self.state_to_index = state_to_index


    def forward(self, sample):
        T = len(sample)
        N = self.num_states
        NC = N * N

        alpha = np.zeros((T, NC))

        for y in range(N):
            for z in range(N):
                idx = self.compound_index[(y, z)]
                alpha[1, idx] = (
                    self.pi_matrix[y]
                    * self.A_matrix[self.compound_index[(y, z)], z]
                    * self.B_matrix[z, sample[1]]
                )

        for t in range(2, T):
            for x in range(N):
                for y in range(N):
                    prev_idx = self.compound_index[(x, y)]
                    a_prev = alpha[t-1, prev_idx]

                    if a_prev == 0:
                        continue

                    for z in range(N):
                        curr_idx = self.compound_index[(y, z)]
                        alpha[t, curr_idx] += (
                            a_prev
                            * self.A_matrix[prev_idx, z]
                            * self.B_matrix[z, sample[t]]
                        )

        return alpha

    
    def backward(self, sample):
        T = len(sample)
        N = self.num_states
        NC = N * N

        beta = np.zeros((T, NC))
        beta[T-1, :] = 1.0

        for t in range(T-2, -1, -1):
            for x in range(N):
                for y in range(N):
                    idx = self.compound_index[(x, y)]

                    for z in range(N):
                        next_idx = self.compound_index[(y, z)]

                        beta[t, idx] += (
                            self.A_matrix[idx, z]
                            * self.B_matrix[z, sample[t+1]]
                            * beta[t+1, next_idx]
                        )

        return beta

    def compute_gamma(self, alpha, beta):
        T, NC = alpha.shape
        gamma = np.zeros_like(alpha)

        for t in range(T):
            eps = 1e-12
            Z = np.sum(alpha[t] * beta[t])
            if Z == 0:
                Z = eps
            gamma[t] = (alpha[t] * beta[t]) / Z

        return gamma
    
    def compute_xi(self, alpha, beta, sample):
        T, NC = alpha.shape
        N = self.num_states

        xi = np.zeros((T-1, NC, N))
        eps = 1e-12  

        for t in range(T-1):
            for idx in range(NC):
                x = idx // N
                y = idx % N

                for z in range(N):
                    next_idx = self.compound_index.get((y, z))
                    if next_idx is None:
                        continue

                    # cálculo da probabilidade conjunta para este par (idx, z)
                    xi[t, idx, z] = alpha[t, idx] * self.A_matrix[idx, z] * self.B_matrix[z, sample[t+1]] * beta[t+1, next_idx]

            # Não tirar essa parte
            norm_factor = np.sum(xi[t])
            if norm_factor == 0:
                norm_factor = eps
            xi[t] /= norm_factor

        return xi

    
    def update_pi(self, gamma):
        N = self.num_states
        pi = np.zeros(N)

        for idx in range(N*N):
            y = idx % N
            pi[y] += gamma[0, idx]

        self.pi_matrix = pi / pi.sum()

    def update_A(self, gamma, xi):
        NC, N = self.A.shape

        for idx in range(NC):
            denom = gamma[:-1, idx].sum()
            if denom == 0:
                continue
            for z in range(N):
                self.A_matrix[idx, z] = xi[:, idx, z].sum() / denom

    def update_B(self, gamma, sample):
        N = self.num_states
        V = self.vocab_size

        B_num = np.zeros((N, V))
        B_den = np.zeros(N)

        for t, w in enumerate(sample):
            for idx in range(N*N):
                y = idx % N
                B_num[y, w] += gamma[t, idx]
                B_den[y] += gamma[t, idx]

        self.B = B_num / B_den[:, None]

    def train(self, df, num_iter=100, eps=1e-2):
        for ite in range(num_iter):
            print(f"Iteração {ite}")

            A_old = self.A_matrix.copy()
            B_old = self.B_matrix.copy()
            pi_old = self.pi_matrix.copy()

            A_acc = np.zeros_like(self.A_matrix)
            B_acc = np.zeros_like(self.B_matrix)
            pi_acc = np.zeros(self.num_states)

            for _, sent in df.groupby("sent_id"):
                obs = [self.word_to_index[w] for w in sent.form]

                alpha = self.forward(obs)
                beta = self.backward(obs)

                gamma = self.compute_gamma(alpha, beta)
                xi = self.compute_xi(alpha, beta, obs)

                # π
                for idx in range(self.num_states ** 2):
                    pi_acc[idx % self.num_states] += gamma[0, idx]

                # A
                for idx in range(self.num_states ** 2):
                    A_acc[idx] += xi[:, idx].sum(axis=0)

                # B
                for t, w in enumerate(obs):
                    for idx in range(self.num_states ** 2):
                        B_acc[idx % self.num_states, w] += gamma[t, idx]

            self.pi_matrix = pi_acc / pi_acc.sum()
            self.A_matrix = A_acc / A_acc.sum(axis=1, keepdims=True)
            self.B_matrix = B_acc / B_acc.sum(axis=1, keepdims=True)

            dist_A = np.linalg.norm(self.A_matrix - A_old)
            dist_B = np.linalg.norm(self.B_matrix - B_old)
            dist_pi = np.linalg.norm(self.pi_matrix - pi_old)

            total_dist = dist_A + dist_B + dist_pi

            if total_dist < eps:
                print(f"Convergiu após {ite+1} iterações {total_dist:.6f} < eps)")
                break



    def predict(self, df_eval):
        predictions = []

        for _, sent in df_eval.groupby("sent_id"):
            obs = [self.word_to_index.get(w, self.word_to_index['<unk>']) 
                for w in sent.form]

            T = len(obs)
            N = self.num_states
            NC = N * N

            delta = np.zeros((T, NC))
            psi = np.zeros((T, NC), dtype=int)
            for y in range(N):
                for z in range(N):
                    idx = self.compound_index[(y, z)]
                    delta[1, idx] = (
                        self.pi_matrix[y]
                        * self.A_matrix[self.compound_index[(y, z)], z]
                        * self.B_matrix[z, obs[1]]
                    )
                    psi[1, idx] = 0
            for t in range(2, T):
                for x in range(N):
                    for y in range(N):
                        prev_idx = self.compound_index[(x, y)]
                        max_prob = -1
                        max_state = 0

                        for z in range(N):
                            prob = (
                                delta[t-1, prev_idx]
                                * self.A_matrix[prev_idx, z]
                                * self.B_matrix[z, obs[t]]
                            )
                            if prob > max_prob:
                                max_prob = prob
                                max_state = z

                        curr_idx = self.compound_index[(y, max_state)]
                        delta[t, curr_idx] = max_prob
                        psi[t, curr_idx] = prev_idx
            states_sequence = []
            last_state = np.argmax(delta[T-1])
            states_sequence.append(last_state)

            for t in range(T-1, 1, -1):
                last_state = psi[t, last_state]
                states_sequence.append(last_state)

            states_sequence.reverse()
            simple_states = []
            for idx in states_sequence:
                y = (idx // N) % N
                simple_states.append(self.states[y])

            simple_states.insert(0, "START")

            predictions.extend(simple_states)

        return predictions
