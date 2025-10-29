from src.models.smoothing import laplace_smoothing

def test_laplace_smoothing():
    bigram = {'the':{ 'cat': 2}, 'cat': {'sat': 1}}
    unigram = {'the': 3, 'cat': 3, 'sat': 1}
    vocab_size = len(unigram)

    smoothed = laplace_smoothing(bigram, unigram, vocab_size)

    assert smoothed['the']['cat'] == (2 + 1) / (3 + vocab_size)
    assert smoothed['cat']['sat'] == (1 + 1) / (3 + vocab_size)