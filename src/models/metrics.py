from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def metrics(true, pred):
    # Acurácia
    acc = accuracy_score(true, pred)

    # Precisão, recall e F1 (macro ou micro)
    precision = precision_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')

    return acc, precision, recall, f1   
