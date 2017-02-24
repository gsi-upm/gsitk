import pytest
import numpy as np

from tests.preprocess_test import text_df

from gsitk.preprocess import normalize
from gsitk.features import (
    features, utils, sentitext, surface
)


def test_view_features():
    resp = features.view_features(pprint=False)
    expected = [
        "- test_numpy => format: numpy",
        "- test_pickle => format: pickle"
    ]
    assert set(resp) - set(expected) == set()


def test_read_features():
    resp1 = utils.read_features('test_numpy')
    assert (resp1 == np.array([[1, 2], [2, 1]])).all()


def test_save_features():
    name = 'mytest_feats'
    to_save = np.array([[3, 3], [4, 4]])
    utils.save_features(to_save, name)
    saved = utils.read_features(name)
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
