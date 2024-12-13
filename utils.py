from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd


def evaluate(classifier, X, y, verbose = True, threshold = 0.5):
    '''Returns: accuracy, f1, precision, recall'''
    y_pred = (classifier.predict_proba(X)[:, 1] >= threshold)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = 2.0 * precision * recall / (precision + recall)
    
    if verbose:
        print(f'Accuracy: {accuracy}')
        print(f'F1: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
    return accuracy, f1, precision, recall

def remove_outliners(df, features):
    '''
    remove samples outside of [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    '''
    temp_df = df.copy()
    for feature in features:
        Q1 = temp_df[feature].quantile(0.25)
        Q3 = temp_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        temp_df = temp_df[(temp_df[feature] >= (Q1 - 1.5 * IQR)) & (temp_df[feature] <= (Q3 + 1.5 * IQR))]
    return temp_df

def compare(models: tuple[str, any], X, y, threshold):
    df = pd.DataFrame(columns=['Model','Accuracy', 'F1', 'Precision', 'Recall'])
    for name, clf in models:
        accuracy, f1, precision, recall = evaluate(clf, X, y, verbose=False, threshold=threshold)
        new_row = {"Model": name,'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}
        df.loc[len(df)] = new_row
    return df