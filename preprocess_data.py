from collections import defaultdict
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from constants import *


LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = stopwords.words('english')


def install_nltk_dependencies():
    for dependency in NLTK_DEPENDENCIES:
        nltk.download(dependency)


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


def lemmatize_sentence(sentence):
    return [
        LEMMATIZER.lemmatize(token)
        for token in nltk.word_tokenize(sentence.lower())
        if token not in STOP_WORDS and token.isalpha()
    ]


def count_unique_words(sentences):
    word_counts = defaultdict(int)

    for sentence in sentences:
        for lemma in lemmatize_sentence(sentence):
            word_counts[lemma] += 1

    return {
        key: value
        for key, value in word_counts.items()
        if value >= MIN_WORD_ENTRIES
    }


def read_unique_words():
    words = load_variable(WORDS_COUNT_FILE)

    return sorted(
        [
            [key, value]
            for key, value in words.items()
        ],
        key=lambda x: x[1],
        reverse=True
    )


def vectorize_sentence(sentence, unique_words=None):
    if not unique_words:
        unique_words = read_unique_words()

    lemmas = set(lemmatize_sentence(sentence))
    return [
        1 if word in lemmas else 0
        for word, count in unique_words
    ]


def get_x(sentences):
    unique_words = read_unique_words()
    return np.array([
        vectorize_sentence(sentence, unique_words)
        for sentence in sentences
    ])


def main():
    install_nltk_dependencies()
    sentences, y = read_dataset(DATASET_FILE)
    words = count_unique_words(sentences)

    dump_variable(words, WORDS_COUNT_FILE)

    X = get_x(sentences)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    dump_variable(X, X_FILE)
    dump_variable(y, Y_FILE)
    dump_variable(label_encoder, ENCODER_FILE)


if __name__ == '__main__':
    main()
