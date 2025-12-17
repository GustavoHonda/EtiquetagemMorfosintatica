from src.models.bigram import BigramModel
from src.data.loader import conllu_to_df
from src.data.preprocess import insert_start_end_tokens, lower_case, multiword_filter, select_6_columns, substitute_unk
from src.models.markov import MarkovModel

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

    # Process training data
    df_train = conllu_to_df('./data/corpus-train.conllu')
    df_train = multiword_filter(df_train)
    df_train = select_6_columns(df_train)
    df_train = lower_case(df_train)
    df_train_markov = insert_start_end_tokens(df_train)
    
    vocab = df_train_markov['form'].unique().tolist()

    markov = MarkovModel(vocab_size=len(vocab))
    markov.train(df_train_markov)

    exit(0)

def main():
    # print("Testing Bigram Model")
    # test_bigram_model()
    print("\nTesting Markov Model")
    test_markov_model()



if __name__ == "__main__":
    main()