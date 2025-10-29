from conllu import parse_incr
import pandas as pd

def conllu_to_df(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for tokenlist in parse_incr(fh):
            # tokenlist.metadata tem comentários (ex.: sent_id, text)
            sent_id = tokenlist.metadata.get('sent_id') if tokenlist.metadata else None
            text = tokenlist.metadata.get('text') if tokenlist.metadata else None

            for token in tokenlist:
                # token é um dict; IDs podem ser int ou str (1-2, 1.1)
                rows.append({
                    'sent_id': sent_id,
                    'text': text,
                    'id': token.get('id'),
                    'form': token.get('form'),
                    'lemma': token.get('lemma'),
                    'upos': token.get('upostag'),   # 'upostag' key name in conllu
                    'xpos': token.get('xpostag'),
                    'feats': token.get('feats'),
                    'head': token.get('head'),
                    'deprel': token.get('deprel'),
                    'deps': token.get('deps'),
                    'misc': token.get('misc'),
                })
    df = pd.DataFrame(rows)
    return df
