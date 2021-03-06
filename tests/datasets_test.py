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
import sys
import pytest
import pandas as pd

from gsitk.datasets import (
    datasets, sentiment140, vader, pl04 , semeval14, semeval13, imdb, imdb_unsup, \
    sst, multidomain, semeval07, sts
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
    assert sorted(data['polarity'].value_counts().index) == [-2, -1, 0, 1, 2]
    assert (data['text'].apply(len) > 0).all()


def test_multidomain():
    multi = multidomain.Multidomain()
    data = multi.prepare_data(download=False)
    pol_val_count = data['polarity'].value_counts()
    assert pol_val_count.index[0] == 1
    assert pol_val_count.values[0] > 0
    assert (data['text'].apply(len) > 0).all()


def test_semeval07():
    se07 = semeval07.Semeval07()
    data = se07.prepare_data(download=False)
    assert 'valence' in data.columns
    assert len(set(['anger','disgust', 'fear','joy','sadness','surprise']) & \
               set(data.columns)) == 6
    assert set(data['fold'].value_counts().index) == set(['dev','test'])


def test_sts():
    sts_data = sts.STS()
    data = sts_data.prepare_data(download=False)
    pol_val_count = data['polarity'].value_counts()
    assert data.shape[0] > 0
    assert pol_val_count.index.shape[0] > 0
    assert pol_val_count.values.shape[0] > 0

def test_find_datasets(dataset_manager):
    '''Dataset definition files should be discoverable'''
    dm = dataset_manager
    root = os.path.dirname(__file__)
    fake_folder = os.path.join(root, 'fake_datasets')
    fake1_path = os.path.join(fake_folder, 'fake1.yml')
    datasets = dm.find_datasets(fake_folder)
    assert fake1_path in datasets


def test_get_dataset_local_data(dataset_manager):
    '''The dataset manager should be able to load datasets from a given location'''
    dm = dataset_manager
    data_path = os.path.join(os.path.dirname(__file__), 'fake_datasets')
    fake1_path = os.path.join(data_path, 'fake1.yml')

    fake1 = dm.get_dataset(fake1_path, data_path=data_path)

    assert fake1.info

    assert len(fake1.data) == fake1.info['stats']['instances']

def test_get_dataset_global_data(dataset_manager):
    '''
    The dataset manager should be able to load datasets from
    a given location, using data from DATA_FOLDER'''
    dm = dataset_manager
    data_path = os.path.join(os.path.dirname(__file__), 'fake_datasets')
    fake2_path = os.path.join(data_path, 'fake2.yml')

    fake2 = dm.get_dataset(fake2_path)

    assert fake2.info

    assert len(fake2.data) == fake2.info['stats']['instances']

def test_get_dataset_pollutes(dataset_manager):
    '''
    The dataset manager should not pollute the import namespace when loading
    a dataset
    '''
    dm = dataset_manager
    data_path = os.path.join(os.path.dirname(__file__), 'fake_datasets')
    fake1_path = os.path.join(data_path, 'fake1.yml')

    fake1 = dm.get_dataset(fake1_path)

    try:
        import fake1
        raise Exception('The global environment has been poluted')
    except ImportError:
        pass


def test_get_dataset_existing_module(dataset_manager):
    '''
    The dataset manager should be able to load a dataset even if its name collides
    with an existing module.
    '''
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from fake3 import is_dataset

    assert not is_dataset

    dm = dataset_manager
    data_path = os.path.join(os.path.dirname(__file__), 'fake_datasets')
    fake3_path = os.path.join(data_path, 'fake3.yml')

    fake3_dataset = dm.get_dataset(fake3_path)

    assert fake3_dataset.is_dataset
