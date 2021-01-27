#
# Copyright 2021 Grupo de Sistemas Inteligentes, DIT, Universidad Politecnica de Madrid (UPM)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
