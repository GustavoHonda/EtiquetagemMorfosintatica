import pandas as pd

## Parte 1

def select_6_columns(df):
    """Select only the four relevant columns from the dataframe."""
    return df[['sent_id', 'text', 'id','form', 'lemma', 'upos']]

def multiword_filter(df):
    """Remove multiword tokens from the dataframe."""
    return df[df['id'].apply(lambda x: isinstance(x, int))]  # Keep only rows where ID is an integer

def select_2_columns(df):
    """Select only the two relevant columns from the dataframe."""
    return df[['sent_id','form']]

def insert_start_end_tokens(df):
    """Insert <s> and </s> tokens at the beginning and end of each sentence."""
    new_rows = []
    df['text'] = "<s> " + df['text'] + " </s>"

    for sent_id, group in df.groupby('sent_id'):
        text = group['text'].iloc[0]
        len = group.shape[0]
        new_rows.append({'sent_id': sent_id,'text': text , 'id': 0,'form': '<s>'})
        new_rows.extend(group.to_dict('records'))
        new_rows.append({'sent_id': sent_id,'text': text , 'id': len + 1, 'form': '</s>'})
    return pd.DataFrame(new_rows)

def lower_case(df):
    """Convert all forms in the dataframe to lower case."""
    df['text'] = df['text'].str.lower()
    df['form'] = df['form'].str.lower()
    return df

def substitute_unk(df, vocab):
    """Substitute words not in the vocabulary with <unk> token."""
    df['form'] = df['form'].apply(lambda x: x if x in vocab else '<unk>')
    return df

## Parte 2