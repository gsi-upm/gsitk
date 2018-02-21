"""
Normalize text
"""

import string
from nltk import word_tokenize

from gsitk.preprocess.pprocess_twitter import tokenize

noise = set(string.punctuation) - set('¡!¿?,.:')  # > and < are removed also
noise = {ord(c): None for c in noise}

def _normalize_text(text):
        t = tokenize(text)
        t = t.lower().translate(noise)
        return word_tokenize(t)

def normalize_text(data):
    # Tokenize and clean the test
    text_data = data['text'].apply(_normalize_text)
    return text_data

def preprocess(text):
    return _normalize_text(text)
