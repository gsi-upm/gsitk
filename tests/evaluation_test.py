import pytest

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from gsitk.datasets import imdb
from gsitk.evaluation.evaluation import Evaluation
from gsitk.pipe import EvalPipeline


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

def test_simple_eval(pipeline_classifier, test_data):
    # Train the pipeline
    pipeline_classifier.fit(test_data['text'].values,
                            test_data['polarity'].values)

    # Declare evaluation tuples
    ev_tuples = [
        EvalPipeline('mypipeline', pipeline_classifier, 'imdb')
    ]

    # Generate evaluation
    ev = Evaluation(
        datasets={'imdb': test_data},
        features=[],
        models=[],
        tuples=ev_tuples
    )

    ev.evaluate()
    results = ev.results
    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == 1
    assert results.shape[1] > 0
