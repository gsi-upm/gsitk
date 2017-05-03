"""
    Tokenization and string cleaning. Modified from:
    https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
"""

import re

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9().,!?\'\`]", " ", string)
    string = re.sub(r"[0-9]+", " num ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def normalize_text(data):
    text_data = data['text'].apply(clean_str)
    text_data = text_data.str.split(' ')
    return text_data
