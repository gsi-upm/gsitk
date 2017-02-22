import pytest
import numpy as np

from gsitk.features import features, utils


def test_view_features():
    resp = features.view_features(pprint=False)
    expected = [
        "- test_numpy => format: numpy",
        "- test_pickle => format: pickle"
    ]
    assert set(resp) - set(expected) == set()


def test_read_features():
    resp1 = utils.read_features('test_numpy')
    assert (resp1 == np.array([[1,2],[2,1]])).all()


def test_save_features():
    name = 'mytest_feats'
    to_save = np.array([[3,3],[4,4]])
    utils.save_features(to_save, name)
    saved = utils.read_features(name)
    assert (saved == to_save).all()

