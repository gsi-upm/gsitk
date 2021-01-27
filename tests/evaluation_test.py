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
import pytest

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from gsitk.datasets import imdb
from gsitk.evaluation.evaluation import Evaluation
from gsitk.pipe import EvalPipeline, EvalTuple, Features


@pytest.fixture
def single_classifier():
    return SGDClassifier()

@pytest.fixture
def simple_feature_extractor():
    return CountVectorizer()


@pytest.fixture
def pipeline_classifier():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])
    return pipeline

@pytest.fixture
def test_data():
    imdb_data = imdb.Imdb()
    data = imdb_data.prepare_data(download=False)
    data['text'] = data['text'].apply(' '.join)
    data['polarity'] = data['polarity'].astype(int)
    return data 

def test_eval_pipeline_simple(pipeline_classifier, test_data):
    # Train the pipeline
    train_indices = (test_data['fold'] == 'train').values
    pipeline_classifier.fit(test_data['text'].values[train_indices],
                            test_data['polarity'].values[train_indices])
    pipeline_classifier.name = 'mypipeline'

    # Declare evaluation tuples
    ev_tuples = [
        EvalPipeline(pipeline_classifier, 'imdb')
    ]

    # Generate evaluation
    ev = Evaluation(
        datasets={'imdb': test_data},
        features=[],
        tuples=ev_tuples,
        pipelines=None
    )

    ev.evaluate()
    results = ev.results
    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == 1
    assert results.shape[1] > 0


def test_eval_tuple_simple(single_classifier, simple_feature_extractor, test_data):
    classifier = single_classifier
    feature_ext = simple_feature_extractor

    # Train the classifier
    train_indices = (test_data['fold'] == 'train').values
    test_indices = (test_data['fold'] == 'test').values

    X_train = feature_ext.fit_transform(test_data['text'].values[train_indices])
    y_train = test_data['polarity'].values[train_indices] 
    classifier.fit(X_train, y_train)
    classifier.name = 'mylogisticregression'

    X_test = feature_ext.transform(test_data['text'].values[test_indices])
    feats = [Features(name='ngrams__imdb_test', dataset='imdb', values=X_test)]

    ev_tuples = [
        EvalTuple(classifier, 'ngrams__imdb_test', 'imdb')
    ]

    ev = Evaluation(
        datasets={'imdb': test_data},
        features=feats,
        tuples=ev_tuples,
        pipelines=None
    )

    ev.evaluate()
    results = ev.results
    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == 1
    assert results.shape[1] > 0
