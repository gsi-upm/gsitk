"""
Normalize text
"""

import string
from nltk import word_tokenize

from gsitk.preprocess.pprocess_twitter import tokenize


def normalize_text(data):
    noise = set(string.punctuation) - set('¡!¿?,.:')  # > and < are removed also
    noise = {ord(c): None for c in noise}

    def _normalize_text(text):
        t = tokenize(text['text'])
        t = t.lower().translate(noise)
        return word_tokenize(t)

    # Tokenize and clean the test
    text_data = data.apply(_normalize_text, axis=1)
    return text_data