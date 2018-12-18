from cached_property import cached_property
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

from constants import *


class TextEncoder(object):
    def __init__(self):
        self.unique_words = []

    @cached_property
    def lemmatizer(self):
        return WordNetLemmatizer()

    @cached_property
    def stop_words(self):
        return stopwords.words('english')

    def fit(self, X):
        self.unique_words = self._get_sorted_unique_words(X)

    def transform(self, X):
        return np.array([
            self._vectorize_sentence(sentence)
            for sentence in X
        ])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _lemmatize_sentence(self, sentence):
        return [
            self.lemmatizer.lemmatize(token)
            for token in nltk.word_tokenize(sentence.lower())
            if token not in self.stop_words and token.isalpha()
        ]

    def _vectorize_sentence(self, sentence):
        lemmas = set(self._lemmatize_sentence(sentence))
        return [
            1 if word in lemmas else 0
            for word, count in self.unique_words
        ]

    def _count_unique_words(self, X):
        word_counts = defaultdict(int)

        for sentence in X:
            for lemma in self._lemmatize_sentence(sentence):
                word_counts[lemma] += 1

        return {
            key: value
            for key, value in word_counts.items()
            if value >= MIN_WORD_ENTRIES
        }

    def _get_sorted_unique_words(self, X):
        words = self._count_unique_words(X)
        return sorted(
            [
                [key, value]
                for key, value in words.items()
            ],
            key=lambda x: x[1],
            reverse=True
        )
