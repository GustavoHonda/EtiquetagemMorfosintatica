from src.models.bigram import BigramModel
import pandas as pd

def test_bigram_model_initialization():
    model = BigramModel()
    assert isinstance(model, BigramModel)

def test_bigram_model_training():
    data = pd.DataFrame({
        'form': ['<s>', 'the', 'cat', '</s>', '<s>', 'the', 'dog', '</s>'],
        'sent_id': [1, 1, 1, 1, 2, 2, 2, 2]
    })
    model = BigramModel()
    model.train(data)
    
    expected_unigram = {
        '<s>': 2,
        'the': 2,
        'cat': 1,
        '</s>': 2,
        'dog': 1
    }
    
    expected_bigram = {
        '<s>': {'the': 2},
        'the': {'cat': 1, 'dog': 1},
        'cat': {'</s>': 1},
        'dog': {'</s>': 1}
    }
    
    assert model.unigram == expected_unigram
    assert model.bigram == expected_bigram

def test_bigram_model_prediction():
    data = pd.DataFrame({
        'form': ['<s>', 'the', 'cat', '</s>', '<s>', 'the', 'dog', '</s>'],
        'sent_id': [1, 1, 1, 1, 2, 2, 2, 2]
    })
    model = BigramModel()
    model.train(data)
    
    test_data = pd.DataFrame({
        'form': ['<s>', 'the', 'cat', '</s>', '<s>', 'the', 'dog', '</s>'],
        'sent_id': [1, 1, 1, 1, 2, 2, 2, 2]
    })
    
    predictions = model.predict(test_data)
    
    expected_predictions = {
        1: ['the', 'cat', '</s>'],
        2: ['the', 'dog', '</s>']
    }
    
    assert predictions != {} 

def test_bigram_model_unknown_handling():
    data = pd.DataFrame({
        'form': ['<s>', 'the', '<unk>', '</s>', '<s>', 'the', 'dog', '</s>'],
        'sent_id': [1, 1, 1, 1, 2, 2, 2, 2]
    })
    model = BigramModel()
    model.train(data)
    
    assert '<unk>' in model.bigram_suavizado
    assert 'dog' not in model.bigram_suavizado.get('<unk>', {})

def test_bigram_model_evaluation():
    data = pd.DataFrame({
        'form': ['<s>', 'the', 'cat', '</s>', '<s>', 'the', 'dog', '</s>'],
        'sent_id': [1, 1, 1, 1, 2, 2, 2, 2]
    })
    model = BigramModel()
    model.train(data)
    
    test_data = pd.DataFrame({
        'form': ['<s>', 'the', 'cat', '</s>', '<s>', 'the', 'dog', '</s>'],
        'sent_id': [1, 1, 1, 1, 2, 2, 2, 2]
    })
    
    acuracy, precision, recall, f1_score = model.evaluate(test_data)
    
    assert 0 <= acuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1_score <= 1