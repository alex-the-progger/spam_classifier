import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from constants import *
from text_encoder import TextEncoder

for dependency in NLTK_DEPENDENCIES:
    nltk.download(dependency)


LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = stopwords.words('english')


def dump_variable(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)


def load_variable(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def read_dataset(filename):
    df = pd.read_csv(filename, encoding='ISO-8859-1')

    values = df[['v1', 'v2']].values
    sentences = values[:, 1]
    y = values[:, 0]

    return sentences, y


def main():
    X, y = read_dataset(DATASET_FILE)

    text_encoder = TextEncoder()
    X = text_encoder.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    dump_variable(X, X_FILE)
    dump_variable(y, Y_FILE)
    dump_variable(text_encoder, TEXT_ENCODER_FILE)
    dump_variable(label_encoder, ENCODER_FILE)


if __name__ == '__main__':
    main()
