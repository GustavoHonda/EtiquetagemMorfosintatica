from src.data.loader import conllu_to_df

def test_conllu_to_df_train():
    df = conllu_to_df('./data/corpus-train.conllu')
    assert not df.empty
    assert 'sent_id' in df.columns
    assert 'text' in df.columns
    assert 'id' in df.columns
    assert 'form' in df.columns
    assert 'lemma' in df.columns
    assert 'upos' in df.columns

def test_conllu_to_df_test():
    df = conllu_to_df('./data/corpus-test.conllu')
    assert not df.empty
    assert 'sent_id' in df.columns
    assert 'text' in df.columns
    assert 'id' in df.columns
    assert 'form' in df.columns
    assert 'lemma' in df.columns
    assert 'upos' in df.columns

def test_conllu_to_df_dev():
    df = conllu_to_df('./data/corpus-dev.conllu')
    assert not df.empty
    assert 'sent_id' in df.columns
    assert 'text' in df.columns
    assert 'id' in df.columns
    assert 'form' in df.columns
    assert 'lemma' in df.columns
    assert 'upos' in df.columns