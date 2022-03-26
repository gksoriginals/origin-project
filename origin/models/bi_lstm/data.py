from math import nan
from sklearn.model_selection._split import train_test_split
import pandas as pd
import unicodedata
import string

data_path = 'data/dataset.csv'

def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def extract_features_and_labels(path):
    dataset = pd.read_csv(path, encoding='utf-8')
    data = dataset['name'].values
    labels = dataset['nation'].values
    categories = dataset['nation'].unique()
    labels = [categories.tolist().index(x) for x in labels]
    return data, labels, categories

def create_dataset(path):
    data, labels, categories = extract_features_and_labels(path)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test, categories, len(categories)


X_train, y_train, X_test, y_test, categories, n_classes = create_dataset(data_path)


