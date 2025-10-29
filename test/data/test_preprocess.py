from src.data.preprocess import select_6_columns, multiword_filter, select_2_columns, insert_start_end_tokens

def test_6_columns():
    import pandas as pd
    data = {
        'sent_id': [1, 1, 1, 1],
        'text': "The cat is on",
        'id': [1, 2, '3-4', 5],
        'form': ['The', 'cat', 'is', 'on'],
        'lemma': ['the', 'cat', 'be', 'on'],
        'upos': ['DET', 'NOUN', 'AUX', 'ADP'],
        'XPOS': ['DT', 'NN', 'VBZ', 'IN'],
        'FEATS': [None, None, None, None]
    }
    df = pd.DataFrame(data)
    selected_df = select_6_columns(df)
    
    assert list(selected_df.columns) == ['sent_id','text','id', 'form', 'lemma', 'upos']

def test_multiword_filter():
    import pandas as pd
    data = {
        'id': [1, 2, '3-4', 5, '6-7'],
    }
    df = pd.DataFrame(data)
    filtered_df = multiword_filter(df)
    
    assert len(filtered_df) == 3  # Two multiword tokens should be removed
    assert all(isinstance(id_, int) for id_ in filtered_df['id'])  # All ids should be integers

def test_select_2_columns():
    import pandas as pd
    data = {
        'sent_id': [1, 1, 1],
        'id': [1, 2, 3],
        'form': ['The', 'cat', 'sits'],
        'lemma': ['the', 'cat', 'sit'],
        'upos': ['DET', 'NOUN', 'VERB']
    }
    df = pd.DataFrame(data)
    selected_df = select_2_columns(df)
    
    assert list(selected_df.columns) == ['sent_id', 'form']
    assert len(selected_df) == 3

def test_insert_start_end_tokens():
    import pandas as pd
    data = {
        'sent_id': [1, 1, 1, 2, 2],
        'text': ['The cat sits', 'The cat sits', 'The cat sits', 'A dog', 'A dog'],
        'form': ['The', 'cat', 'sits', 'A', 'dog']
    }
    df = pd.DataFrame(data)
    modified_df = insert_start_end_tokens(df)
    
    expected_data = {
        'sent_id': [1, 1, 1, 1, 1, 2, 2, 2, 2],
        'text': ['<s> The cat sits </s>',
                 '<s> The cat sits </s>',
                 '<s> The cat sits </s>',
                 '<s> The cat sits </s>',
                 '<s> The cat sits </s>',
                '<s> A dog </s>',
                '<s> A dog </s>',
                '<s> A dog </s>',
                '<s> A dog </s>',
                ],
        'form': ['<s>', 'The', 'cat', 'sits', '</s>', '<s>', 'A', 'dog', '</s>']
    }
    expected_df = pd.DataFrame(expected_data)
    
    assert len(modified_df) == len(expected_df)
    assert all(modified_df['form'].values == expected_df['form'].values)
    assert all(modified_df['text'].values == expected_df['text'].values)
    assert all(modified_df['sent_id'].values == expected_df['sent_id'].values)