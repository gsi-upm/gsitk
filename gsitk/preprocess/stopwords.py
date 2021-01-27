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
"""
Remove stopwords.
"""

from sklearn.base import TransformerMixin
from nltk.corpus import stopwords


class StopWordsRemover(TransformerMixin):
    def __init__(self, type='nltk', language='english'):
        self.type = type
        self.language = language
        if self.type == 'nltk':
            self.stop_words = stopwords.words(self.language)
        else:
            raise NotImplementedError

    def fit(self, X, y=None, **fit_params):
        return self

    def remove_stopwords(self, tokens, stop_words):
        return ' '.join([tok for tok in tokens if tok not in stop_words])

    def transform(self, X, y=None, **fit_params):
        transformed = []
        for x in X:
            tokens = self.remove_stopwords(x.split(' '), self.stop_words)
            transformed.append(tokens)
        return transformed
