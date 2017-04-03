import pytest
import pandas as pd

from gsitk.datasets import (
    datasets, sentiment140, vader, pl04 , semeval14, semeval13, imdb, imdb_unsup, sst
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
    pl = pl04.Pl04()
    data = pl.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == 1
    assert data['polarity'].value_counts().values[0] == 10
    assert (data['text'].apply(len) > 0).all()
    assert 'fold' in data.columns

def test_semeval13():
    smeval = semeval13.Semeval13()
    data = smeval.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == 0
    assert data['polarity'].value_counts().values[0] == 5
    assert data['polarity'].value_counts().values[1] == 3
    assert data['polarity'].value_counts().values[2] == 2
    assert (data['text'].apply(len) > 0).all()

def test_semeval14():
    smeval = semeval14.Semeval14()
    data = smeval.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == 0
    assert data['polarity'].value_counts().values[0] == 5
    assert data['polarity'].value_counts().values[1] == 3
    assert data['polarity'].value_counts().values[2] == 1
    assert (data['text'].apply(len) > 0).all()
    assert (data['text'] != 'not available').all()

def test_imdb():
    imdb_data = imdb.Imdb()
    data = imdb_data.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == -1
    assert data['polarity'].value_counts().values[0] == 10
    assert data['polarity'].value_counts().values[1] == 10
    assert (data['text'].apply(len) > 0).all()
    assert 'fold' in data.columns

def test_imdb_unsup():
    imdb_data = imdb_unsup.Imdb_unsup()
    data = imdb_data.prepare_data(download=False)
    assert (data['text'].apply(len) > 0).all()


def test_sst():
    sentitree = sst.Sst()
    data = sentitree.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == 1
    assert data['polarity'].value_counts().values[0] == 7
    assert data['polarity'].value_counts().values[1] == 3
    assert (data['text'].apply(len) > 0).all()
