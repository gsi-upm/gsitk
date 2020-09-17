import os
import pytest
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Doc2Vec

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
    assert isinstance(model.model.vocab, dict)
    assert len(model.model.vocab) > 0

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

def check_features(feats):
    assert feats.shape[0] > 0
    assert feats.shape[1] > 0
    assert feats.min() != 0
    assert feats.max() != 0

def test_simon(embedding_model, mock_lexicon, mock_input):
    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=2, embedding=embedding_model)
    check_features(model.fit_transform(mock_input))

    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=3, embedding=embedding_model)
    check_features(model.fit_transform(mock_input))

    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=2, wordnet_metric='li')
    check_features(model.fit_transform(mock_input))

    model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=2, wordnet_metric='wpath')
    check_features(model.fit_transform(mock_input))

def test_simon_pipeline(embedding_model, mock_lexicon, mock_input, mock_labels):
    simon_model = simon.Simon(lexicon=mock_lexicon, n_lexicon_words=2, embedding=embedding_model)
    model = simon.simon_pipeline(simon_transformer=simon_model, percentile=50)
    check_features(model.fit_transform(mock_input, mock_labels))
