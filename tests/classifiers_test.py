import pytest
import numpy as np

from gsitk.classifiers import VaderClassifier, LexiconSum

@pytest.fixture
def sentiment_text():
    with open('tests/data/test_dataset.txt') as f:
        text = f.readlines()
    return np.array(text)

def test_vader(sentiment_text):
    clf = VaderClassifier()
    preds = clf.predict(sentiment_text)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 2
    assert (preds == [0, 0]).all()

def test_lexicon_sum(sentiment_text):
    lexicon = {
        'good': 1,
        'bad': -1,
        'happy': 0.9,
        'sad': -1,
        'mildly': -0.1,
        'happy!': 1,
    }
    ls = LexiconSum(lexicon)
    sentiment_text = [sentence.split(' ') for sentence in sentiment_text]
    preds = ls.predict(sentiment_text)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 2
    assert (preds == [0, 1]).all()
