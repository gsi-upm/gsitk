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
import pytest
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Doc2Vec
from sklearn.pipeline import Pipeline

from tests.preprocess_test import text_df

from gsitk.preprocess import normalize
from gsitk.features import (
    features, utils, word2vec, sswe, doc2vec, simon
)


def test_view_features():
    resp = features.view_features(pprint=False)
    expected = [
        "- test_numpy => format: numpy",
        "- test_pickle => format: pickle"
    ]
    for i in expected:
        assert i in resp


def test_read_features():
    resp1 = features.load_features('test_numpy')
    assert (resp1 == np.array([[1, 2], [2, 1]])).all()


def test_save_features():
    name = 'mytest_feats'
    to_save = np.array([[3, 3], [4, 4]])
    features.save_features(to_save, name)
    saved = features.load_features(name)
    assert (saved == to_save).all()


@pytest.fixture
def norm_text(text_df):
    return normalize.normalize_text(text_df)


@pytest.fixture
def embedding_model():
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'data/w2v_model')
    model = KeyedVectors.load_word2vec_format(path)
    return model

def test_word2vec(norm_text, embedding_model):
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'data/w2v_model')
    model = word2vec.Word2VecFeatures(w2v_model_path=path, w2v_format='google_txt')
    assert isinstance(model.model, Word2Vec) or \
        isinstance(model.model, KeyedVectors)
    assert isinstance(model.model.key_to_index, dict)
    assert isinstance(model.model.index_to_key, list)
    assert len(model.model.key_to_index) > 0
    assert len(model.model.index_to_key) > 0

    x = model.transform(norm_text)
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, model.size)
    assert (x != 0).all()

def test_word2vec_load(norm_text, embedding_model):
    model = word2vec.Word2VecFeatures(model=embedding_model)
    assert isinstance(model.model, Word2Vec) or \
        isinstance(model.model, KeyedVectors)
    assert isinstance(model.model.key_to_index, dict)
    assert isinstance(model.model.index_to_key, list)
    assert len(model.model.key_to_index) > 0
    assert len(model.model.index_to_key) > 0

    x = model.transform(norm_text)
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, model.size)
    assert (x != 0).all()


def test_sswe(norm_text):
    model = sswe.SSWE(download=False)
    assert isinstance(model.model, dict)
    assert len(model.model) > 0

    x = model.transform(norm_text)
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, model.size)
    assert (x != 0).all()


def test_doc2vec(norm_text):
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'data/d2v_model')
    model = doc2vec.Doc2VecFeatures(d2v_model_path=path)
    assert isinstance(model.model, Doc2Vec)
    assert isinstance(model.model.vector_size, int)
    assert model.model.vector_size > 0

    x = model.transform(norm_text)
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, model.size)
    assert (x != 0).all()


@pytest.fixture
def mock_lexicon():
    lexicon = [
        ['my', 'number', 'you'],
        ['user', 'i', 'to']
    ]
    return lexicon

@pytest.fixture
def mock_input():
    input_data = [
        ['repeat', 'number', 'allcaps','you',  'and'],
        ['is', 'it', 'for', 'user', 'i', 'to']
    ]
    return input_data

@pytest.fixture
def mock_labels():
    return [1, 0]

def check_features(feats, zeros=False):
    assert feats.shape[0] > 0
    assert feats.shape[1] > 0
    if not zeros:
        assert feats.min() != 0
        assert feats.max() != 0
    else:
        assert feats.min() == 0
        assert feats.max() == 0

def test_simon(embedding_model, mock_lexicon, mock_input):
    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=2, embedding=embedding_model)
    check_features(model.fit_transform(mock_input))

    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=3, embedding=embedding_model)
    check_features(model.fit_transform(mock_input))

    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=2, wordnet_metric='li')
    check_features(model.fit_transform(mock_input))

    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=2, wordnet_metric='wpath')
    check_features(model.fit_transform(mock_input))

def test_simon_oov(embedding_model, mock_lexicon):
    oov_input = [['sad', 'cat'], ['happy', 'dog']]
    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=3, embedding=embedding_model)
    check_features(model.fit_transform(oov_input), zeros=True)

def test_simon_pipeline(embedding_model, mock_lexicon, mock_input, mock_labels):
    simon_model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=2, embedding=embedding_model)
    model = simon.simon_pipeline(simon_transformer=simon_model, percentile=50)
    check_features(model.fit_transform(mock_input, mock_labels))

def test_simon_sklearn_pipeline(embedding_model, mock_lexicon, mock_input):
    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=3, embedding=embedding_model)
    pipeline = Pipeline([
        ('simon', model),
    ])
    pipeline.fit(mock_input)
    check_features(pipeline.transform(mock_input))

    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=3, embedding=embedding_model)
    pipeline = Pipeline([
        ('simon', model),
    ])
    check_features(pipeline.fit_transform(mock_input))

def test_simon_sklearn_pipeline_steps(embedding_model, mock_lexicon, mock_input):
    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=3, embedding=embedding_model)
    print(model.pooling)
    pipeline = Pipeline([
        ('simon', model),
    ])
    pipeline.fit(mock_input, [1, 0])

    print(pipeline.steps)

    steps = pipeline.steps
    assert isinstance(steps, list)
    assert len(steps) > 0

