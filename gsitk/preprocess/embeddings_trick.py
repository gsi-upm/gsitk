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
Implements the embeddings trick. Substitutes a word outside of vocabulary for
the most close to it. It provides an enhancement of vocabulary.
"""

from sklearn.base import TransformerMixin

from gsitk.features.word2vec import Word2VecFeatures

class EmbeddingsTricker(TransformerMixin):
    def __init__(self, model_path, w2v_format, vocabulary):
        w2v = Word2VecFeatures(model_path, w2v_format)
        self.wv = w2v.model 
        self.vocab = set(vocabulary)

    def fit(self, X, y=None):
        return self

    def closest_word(self, w):
        # get closest word in embedding space
        return self.wv.most_similar(w, topn=1)[0][0]

    def transform(self, X, y=None):
        for i, x_i in enumerate(X):
            for j, word in enumerate(x_i):
                if word not in self.vocab:
                    closest = self.closest_word(word)
                    X[i][j] = closest
        
        return X
