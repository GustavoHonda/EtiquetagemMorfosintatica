def laplace_smoothing(bigram, unigram, vocab_size):
    smoothed_bigram = {}
    for w1 in bigram:
        smoothed_bigram[w1] = {}
        for w2 in bigram[w1]:
            count_w1_w2 = bigram[w1][w2]
            count_w1 = unigram.get(w1, 0)
            smoothed_prob = (count_w1_w2 + 1) / (count_w1 + vocab_size)
            smoothed_bigram[w1][w2] = smoothed_prob
    return smoothed_bigram
