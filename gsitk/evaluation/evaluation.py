"""
Evaluation of classifiers on datasets.
"""

import logging
from collections import namedtuple
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class Evaluation():
    """
    Models an evaluation process.
    """
    def __init__(self, datasets, features, models, tuples):
        self.Dataset = namedtuple('Dataset', ['name','dataframe'])
        self.Prediction = namedtuple('Prediction', ['name',
                                                    'dataset',
                                                    'features',
                                                    'model',
                                                    'values'])
        self.Label = namedtuple('Label', ['name', 'values'])
        self.Metric = namedtuple('Metric', ['type',
                                            'dataset',
                                            'features',
                                            'model',
                                            'value'])
        self.metrics_defs = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        self.models = self._models_setup(models)
        self.features = self._features_setup(features)
        self.datasets = self._datasets_setup(datasets)
        self.tuples = tuples
        self.predictions = dict()
        self.labels = self._labels_setup()
        self.results = self._results_setup()

    def _results_setup(self):
        columns = ['Dataset', 'Features', 'Model']
        columns.extend(self.metrics_defs)
        return  pd.DataFrame(columns=columns)

    def _datasets_setup(self, datasets):
        _datasets = dict()
        for name, dataframe in datasets.items():
            _datasets[name] = self.Dataset(name, dataframe)
        return _datasets

    def _models_setup(self, models):
        _models = dict()
        for model in models:
            _models[model.name] = model
        return _models

    def _features_setup(self, features):
        _features = dict()
        for features_ in features:
            _features[features_.name] = features_
        return _features

    def _labels_setup(self):
        _labels = dict()
        for dataset_name, dataset_obj in self.datasets.items():
            _labels[dataset_name] = dataset_obj.dataframe['polarity'].values
        return _labels

    def _select_metric_call(self, metric_name):
        if metric_name == 'Accuracy':
            return accuracy_score
        elif metric_name == 'Recall':
            return recall_score
        elif metric_name == 'Precision':
            return precision_score
        elif metric_name == 'F1-Score':
            return f1_score

    def evaluate_model(self, features_name, model_name):
        """Predict from features."""
        logger.debug('Model {} predicting from features {}'.format(model_name, features_name))
        _features = self.features[features_name]
        _model = self.models[model_name]
        predictions = _model.classifier.predict(_features.values)
        prediction = self.Prediction('{}__{}'.format(model_name, features_name),
                                     _features.dataset,
                                     _features.name,
                                     _model.name,
                                     predictions)
        self.predictions['{}__{}'.format(model_name, features_name)] = prediction

    def evaluate(self):
        """
        Evaluate all models on all datasets.
        Tuples represent the (classifier_name, train_features_name, test_labels_name)
        pairs that are going to be evaluated.
        """

        for tuple in self.tuples:
            self.evaluate_model(tuple.features, tuple.classifier)

        for prediction in self.predictions.values():
            labels = self.labels[prediction.dataset]
            preds = prediction.values
            test_dataset_name = prediction.dataset
            features_name = prediction.features
            model_name = prediction.model

            scores = list()
            for metric in self.metrics_defs:
                func = self._select_metric_call(metric)
                scores.append(func(labels, preds))

            i = self.results.shape[0]
            if i == 0:
                i = 0
            else:
                i += 1

            result = [test_dataset_name, features_name, model_name]
            result.extend(scores)

            self.results.loc[i, :] = result
