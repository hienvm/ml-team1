from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def evaluate(classifier, X, y, verbose = True):
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    if verbose:
        print(f'Accuracy: {acc}')
        print(f'F1: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
    return acc, f1, precision, f1