import os
import pytest
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Doc2Vec

from tests.preprocess_test import text_df

from gsitk.preprocess import normalize
from gsitk.features import (
    features, utils, sentitext, surface, word2vec, sswe, doc2vec
)


def test_view_features():
    resp = features.view_features(pprint=False)
    expected = [
        "- test_numpy => format: numpy",
        "- test_pickle => format: pickle"
    ]
    assert set(resp) - set(expected) == set()


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


def test_sentitext(norm_text):
    st = sentitext.prepare_data(norm_text)
    assert len(st) > 0
    for st_i in st:
        assert len(st_i.keys()) > 0
        assert isinstance(st_i['total_score'], float)


def test_surface(norm_text):
    x = surface.transform(norm_text)
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, 12)
    assert sum(x[0]) == -4.0
    assert sum(x[1]) == 4.0


def test_word2vec(norm_text):
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'data/w2v_model')
    model = word2vec.Word2VecFeatures(w2v_model_path=path,
                                      w2v_format='google_txt')
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
