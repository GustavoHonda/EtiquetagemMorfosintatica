from src.models.bigram import BigramModel
from src.data.loader import conllu_to_df
from src.data.preprocess import insert_start_end_tokens, lower_case, multiword_filter, select_2_columns, select_6_columns, substitute_unk
from src.models.markov import MarkovModel
from src.models.markov_lib import HMM
from src.models.metrics import metrics
import pandas as pd

def test_bigram_model():
     # Process training data
    df_train = conllu_to_df('./data/corpus-train.conllu')
    df_train = multiword_filter(df_train)
    df_train = select_6_columns(df_train)
    df_train = lower_case(df_train)
    df_train_bigram = insert_start_end_tokens(df_train)
    
    bigram = BigramModel()
    bigram.train(df_train_bigram)

    # Process test data
    df_test = conllu_to_df('./data/corpus-dev.conllu')
    df_test = multiword_filter(df_test)
    df_test = select_6_columns(df_test)
    df_test = lower_case(df_test)
    df_test_bigram = insert_start_end_tokens(df_test)
    vocab = bigram.unigram.keys()
    df_test_bigram = substitute_unk(df_test_bigram, vocab)

    result = bigram.predict(df_test_bigram)

    for sent_id, predicted_words in result.items():
        print(f"Sent ID: {sent_id}")
        print(f"Predicted Words: {predicted_words}")
        print(f"Expected Words: {df_test.loc[df_test['sent_id'] == sent_id, 'form'].tolist()}")
        break
    
    df_eval = conllu_to_df('./data/corpus-test.conllu')
    df_eval = multiword_filter(df_eval)
    df_eval = select_6_columns(df_eval)
    df_eval = lower_case(df_eval)
    df_eval = insert_start_end_tokens(df_eval)
    vocab = bigram.unigram.keys()
    df_eval = substitute_unk(df_eval, vocab)
        
    metrics, gold_count = bigram.evaluate(df_eval)

    sum_metrics = {}
    for key in metrics:
        sum_metrics[key] = sum(metrics[key]) / len(metrics[key])

    print(f"Acurácia: {sum_metrics['acuracy']}, Precisão: {sum_metrics['precision']}, Cobertura: {sum_metrics['recall']}, Medida F1: {sum_metrics['f1_score']}")

    gold_count = sorted(gold_count.items(), key=lambda x: x[1], reverse=True)
    print("Palavras Preditas Corretamente (Top 10):")
    for sent, count in gold_count[:10]:
        print(f"Palavra: {sent}, Count: {count}")


def test_markov_model():
    df_train = conllu_to_df('./data/corpus-train.conllu')
    df_train = multiword_filter(df_train)
    df_train = select_2_columns(df_train)
    df_train = lower_case(df_train)
    # df_train = insert_start_end_tokens(df_train, no_id=True)
    
    

    unk_row = pd.DataFrame({
        'sent_id': ['TRAIN_UNK_SENT'],
        'text': ['<unk> placeholder sentence'],
        'form': ['<unk>'],
        'upos': ['X'] 
    })
    df_train = pd.concat([df_train, unk_row], ignore_index=True)
    vocab = df_train['form'].unique().tolist()
    states = df_train['upos'].unique().tolist()


    df_dev = conllu_to_df('./data/corpus-dev.conllu')
    df_dev = multiword_filter(df_dev)
    df_dev = select_2_columns(df_dev)
    df_dev = lower_case(df_dev)
    # df_dev = insert_start_end_tokens(df_dev, no_id=True)
    df_dev = substitute_unk(df_dev, vocab)

    df_eval = conllu_to_df('./data/corpus-test.conllu')
    df_eval = multiword_filter(df_eval)
    df_eval = select_2_columns(df_eval)
    df_eval = lower_case(df_eval)
    # df_eval = insert_start_end_tokens(df_eval, no_id=True)
    df_eval = substitute_unk(df_eval, vocab)
    
    

    vocab = df_train['form'].unique().tolist()
    states = df_train['upos'].unique().tolist()
    print(f"Vocab size: {len(vocab)}"
          f"\nStates size: {len(states)}")
    
    print(f"States: {states}")
    
    model = MarkovModel(df_train)
    # print(f"Initial Probabilities (pi): {model.pi_matrix}"
    #       f"\nTransition Probabilities (A) sample: {dict(list(model.A_dict.items())[:1])}"
    #       f"\nEmission Probabilities (B) sample: {dict(list(model.B_dict.items())[:2])}")

    model.train( df=df_dev, num_iter=5, eps=1e-2)
    prediction = model.predict(df_eval=df_eval)

    print(f"Predictions sample: {prediction[:20]}")
    true_values = df_eval.head(20)['upos'].tolist()
    print(f"True values sample: {true_values}")


    mask = df_eval["upos"] != "START"

    



    metrics_result = metrics(df_eval.loc[mask, "upos"].tolist(), [p for p, m in zip(prediction, mask) if m])
    print(f"Metrics (Library HMM) - Acurácia: {metrics_result[0]}, Precisão: {metrics_result[1]}, Cobertura: {metrics_result[2]}, Medida F1: {metrics_result[3]}")

    df_analysis = df_eval.copy()
    df_analysis["y_true"] = df_eval["upos"].tolist()
    df_analysis["y_pred"] = prediction

    # Marcar erro
    df_analysis["error"] = df_analysis["y_true"] != df_analysis["y_pred"]

    stats = (
        df_analysis
        .groupby("form") 
        .agg(
            count=("form", "size"),
            errors=("error", "sum")
        )
    )

    # Taxa de erro
    stats["error_rate"] = stats["errors"] / stats["count"]

    # Filtrar palavras muito raras
    stats = stats[stats["count"] >= 10]

    # Top 10 mais difíceis
    top10_hardest = stats.sort_values("error_rate", ascending=False).head(10)

    print("\n🔴 Top 10 palavras mais difíceis de predizer UPOS:")
    print(top10_hardest)


    exit(0)


def test_hmm_library():


    df_dev = conllu_to_df('./data/corpus-dev.conllu')
    df_dev = multiword_filter(df_dev)
    df_dev = select_2_columns(df_dev)
    df_dev = lower_case(df_dev)
    df_dev = insert_start_end_tokens(df_dev, no_id=True)
    

    unk_row = pd.DataFrame({
        'sent_id': ['TRAIN_UNK_SENT'],
        'text': ['<unk> placeholder sentence'],
        'form': ['<unk>'],
        'upos': ['X'] 
    })
    df_dev = pd.concat([df_dev, unk_row], ignore_index=True)
    vocab = df_dev['form'].unique().tolist()
    states = df_dev['upos'].unique().tolist()


    df_train = conllu_to_df('./data/corpus-train.conllu')
    df_train = multiword_filter(df_train)
    df_train = select_2_columns(df_train)
    df_train = lower_case(df_train)
    df_train = insert_start_end_tokens(df_train, no_id=True)
    df_train = substitute_unk(df_train, vocab)

    df_eval = conllu_to_df('./data/corpus-test.conllu')
    df_eval = multiword_filter(df_eval)
    df_eval = select_2_columns(df_eval)
    df_eval = lower_case(df_eval)
    df_eval = insert_start_end_tokens(df_eval, no_id=True)
    df_eval = substitute_unk(df_eval, vocab)


    # Library HMM
    model = HMM( df_dev=df_dev)
    model.train( df_train=df_train)
    prediction = model.predict( df_eval=df_eval)

    print(f"Predictions sample: {prediction[:20]}")
    true_values = df_eval.head(20)['form'].tolist()
    print(f"True values sample: {true_values}")

    metrics_result = metrics(df_eval['form'].tolist(), prediction)
    print(f"Metrics (Library HMM) - Acurácia: {metrics_result[0]}, Precisão: {metrics_result[1]}, Cobertura: {metrics_result[2]}, Medida F1: {metrics_result[3]}")


def main():
    print("Testing Bigram Model")
    test_bigram_model()
    # print("\nTesting Markov Model")
    # test_markov_model()
    # print("\nTesting HMM Library Model")
    # test_hmm_library()



if __name__ == "__main__":
    main()