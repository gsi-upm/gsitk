import pytest
import pandas as pd

from gsitk.datasets import (
    datasets, sentiment140, vader
)


@pytest.fixture
def dataset_manager():
    return datasets.DatasetManager()


def test_dataset_manager(dataset_manager):
    dm = dataset_manager 
    assert isinstance(dm.infos, dict)
    assert len(dm.infos) > 0
    assert isinstance(dm.datasets, dict)
    assert len(dm.datasets) > 0
    assert set(dm.infos.keys()) == set(dm.datasets.keys())


def test_dataset_manager_prepare_datasets(dataset_manager):
    dm = dataset_manager
    data = dm.prepare_datasets(download=False)
    assert isinstance(data, dict)
    assert len(data) > 0
    for d in data.values():
        assert isinstance(d, pd.DataFrame)
        assert d.shape[0] > 0 and d.shape[1] > 0


def test_sentiment140():
    sent140 = sentiment140.Sentiment140()
    data = sent140.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == -1
    assert data['polarity'].value_counts().values[0] == 10
    assert (data['text'].apply(len) > 0).all()


def test_vader():
    vad = vader.Vader()
    data = vad.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == 1
    assert data['polarity'].value_counts().values[0] == 10
    assert (data['text'].apply(len) > 0).all()

def test_pl04():
    vad = vader.Vader()
    data = vad.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == 1
    assert data['polarity'].value_counts().values[0] == 10
    assert (data['text'].apply(len) > 0).all()
