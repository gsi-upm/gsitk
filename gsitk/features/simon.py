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
import os
import numpy as np
import itertools
from collections import Counter
from nltk.corpus import stopwords, wordnet
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from gsitk.features.wn_similarity import WordNetSimilarity

class Simon(TransformerMixin, BaseEstimator):

    def __init__(self, lexicon, n_lexicon_words=250, embedding=None,
                 wordnet_metric=None, wordnet_cache=False, pooling=np.max, weighting=False,
                 remove_stopwords=False, lex_values=None):

        self._lexicon_split = lexicon
        self.n_lexicon_words = n_lexicon_words
        self._pooling = pooling if callable(pooling) else None
        if embedding is None:
            assert wordnet_metric is not None
        else:
            assert wordnet_metric is None
        self.wordnet_metric = wordnet_metric
        self.wordnet_cache = wordnet_cache
        self.embedding = embedding
        self.remove_stopwords = remove_stopwords
        self.weighting = weighting
        if self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = None
        self.lex_values = lex_values
        self.sentiment_weights = True if self.lex_values is not None else False

    def _prepare_lexicon(self):
        """
        Expects a lexicon as a list.
        """

        count = Counter(itertools.chain.from_iterable(self._text))
        count_sum = np.sum(list(count.values()))
        
        lexicon = list()
        for lexicon_split in self._lexicon_split:
            lex_words = sorted([(word_i, count[word_i]) for word_i in lexicon_split],
                                key=lambda x: x[1], reverse=True)
            lex_words = lex_words[:self.n_lexicon_words] 
            lexicon.append(lex_words)
        self.lexicon = list(itertools.chain.from_iterable(lexicon))
        self.word_ws = [lex_i[1]/count_sum  for lex_i in self.lexicon]
        self.lex_words = [word[0] for word in self.lexicon]

        assert len(self.word_ws) == len(self.lex_words)

        if self.embedding is None:
            all_lemmas = set(wordnet.all_lemma_names())
            self.all_lemmas = sorted(all_lemmas)
            lex_words = list()
            lex_ws = list()
            for i, lex_word_i in enumerate(self.lex_words):
                if not lex_word_i in all_lemmas:
                    continue
                lex_words.append(lex_word_i)
                lex_ws.append(self.word_ws[i])
            self.lex_words = lex_words
            self.word_ws = lex_ws

        if self.lex_values is not None:
            self.lex_values = np.array([self.lex_values[word] for word in self.lex_words])

    def _generate_wordnet_cache(self):
        cache = np.zeros((len(self.all_lemmas), len(self.lex_words)))
        for index, vocab_word in enumerate(self.all_lemmas):
            cache[index] = np.array([self.wns.word_similarity(vocab_word, lex_word) \
                                 for lex_word in self.lex_words])
        self.cache = cache
        self.S = [self.lex_values[i] for i, word in enumerate(self.lex_words)]
        self.W = [self.word_ws[i] for i, word in enumerate(self.lex_words)]

    def _load_wordnet(self):
        wns = WordNetSimilarity()
        self.wns = wns

        name = 'simM_cache.npy'
        if not self.wordnet_cache:
            self._generate_wordnet_cache()
        else:
            if not os.path.exists(name):
                self._generate_wordnet_cache()
                with open(name, 'wb') as f:
                    np.save(f, self.cache)
            else:
                with open(name, 'rb') as f:
                    self.cache = np.load(f)

        self.cache_index = {w: i for i, w in enumerate(self.all_lemmas)}

    def _load_embeddings(self):
        L, W, S = list(), list(), list()
        indexes = list()

        for i, word in enumerate(self.lex_words):
            try:
                v = self.embedding[word]
                indexes.append(i)
            except KeyError:
                continue
            L.append(v)
            W.append(self.word_ws[i])

        if self.sentiment_weights:
            S = [self.lex_values[i] for i in indexes]

        self.L = np.array(L)
        self.W = np.array(W)
        self.S = np.array(S)

    def _extract_embeddings(self, x):
        V = list()
        for i, word in enumerate(x):
            try:
                v = self.embedding[word]
            except KeyError:
                continue
            V.append(v)
        return np.array(V)

    def _compute_with_embeddings(self, x):
        M = list()
        for i, x_i in enumerate(x):
            V = self._extract_embeddings(x_i)
            if len(V) > 0: # V words may be outside embedding vocabulary
                M_i = np.dot(V, self.L.T)
            else:
                M_i = np.zeros((self.L.T.shape[1],))
                M.append(M_i)
                continue

            if self.weighting:
                M_i = self.W * M_i
            if self._pooling is None:
                M.append(M_i)
            else:
                M.append(self._pooling(M_i, axis=0))
        return np.array(M)

    def _fetch_from_cache(self, word):
        index = self.cache_index.get(word, None)
        if index is None:
            return None
        return self.cache[index]

    def _compute_with_wordnet(self, x):
        M = list()
        for i, x_i in enumerate(x):
            M_i = np.zeros((len(x_i), len(self.lex_words)))
            for j, word in enumerate(x_i):
                m_i_j = [self.wns.word_similarity(word, lex_word, self.wordnet_metric) \
                         for lex_word in self.lex_words]
                #m_i_j = self._fetch_from_cache(word)
                M_i[j] = np.array(m_i_j)

            if self._pooling is None:
                M.append(M_i)
            else:
                M.append(self._pooling(M_i, axis=0))
        return np.array(M)

    def _remove_stopwords(self, x):
        return [word for word in x if word not in self.stopwords]

    def fit(self, x, y=None):
        self._text = x
        self._prepare_lexicon()
        if self.embedding is not None:
            self._load_embeddings()
        else:
            wns = WordNetSimilarity()
            self.wns = wns

            #self._load_wordnet()
        return self

    def transform(self, x):
        if self.remove_stopwords:
            x = map(self._remove_stopwords, x)
            x = np.array(list(x))

        if self.embedding is not None:
            M = self._compute_with_embeddings(x)
        else:
            M = self._compute_with_wordnet(x)

        if self.sentiment_weights:
            return self.S * M

        return M


def simon_pipeline(simon_transformer, percentile):
    return Pipeline([
        ('simon', simon_transformer),
        ('scale', MinMaxScaler(feature_range=(-1,1))),
        ('percent', feature_selection.SelectPercentile(feature_selection.f_classif, percentile=percentile)),
    ])
