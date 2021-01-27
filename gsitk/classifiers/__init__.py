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
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import ClassifierMixin

polarity_transform = {0: 1, 1: 0, 2: -1}

class VaderClassifier(ClassifierMixin):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X, y=None):
        y_hat = np.zeros((len(X)))
        for i, x_i in enumerate(X):
            pol = self.sid.polarity_scores(x_i)
            pred = np.array([pol['pos'], pol['neu'], pol['neg']]).argmax()
            y_hat[i] = polarity_transform[pred]

        return np.array(y_hat)


class LexiconSum(ClassifierMixin):
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def fit(self, X, y=None):
        return self

    def polarity_normalize(self, pol):
        if pol > 0:
            pol = 1
        elif pol < 0:
            pol = -1
        else:
            pol = 0
        return pol

    def predict(self, X, y=None):
        y_hat = np.zeros((len(X),))
        for i, x_i in enumerate(X):
            word_pols = [self.lexicon.get(word, 0) for word in x_i]
            pol = np.sum(word_pols)
            y_hat[i] = self.polarity_normalize(pol)
        return y_hat
            