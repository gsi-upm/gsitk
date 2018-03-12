import pytest
import numpy as np

from gsitk.classifiers.vader import VaderClassifier

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