"""
Evaluation of classifiers on datasets.
"""

import logging
from collections import namedtuple
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import cross_val_score

from gsitk.pipe import EvalTuple, CrossEvalTuple

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

        self.metrics_defs = ['accuracy', 'precision', 'recall',
                             'f1', 'f1_weighted', 'f1_micro', 'f1_macro']

        self.folds_defs = ['train', 'dev', 'test']

        self.models = self._models_setup(models)
        self.features = self._features_setup(features)
        self.datasets = self._datasets_setup(datasets)
        self.tuples = tuples
        self.predictions = dict()
        self.labels = self._labels_setup()
        self.results = self._results_setup()

    def _results_setup(self):
        columns = ['Dataset', 'Features', 'Model', 'CV']
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
        if metric_name == 'accuracy':
            return accuracy_score
        elif metric_name == 'recall':
            return recall_score
        elif metric_name == 'precision':
            return precision_score
        elif metric_name == 'f1':
            return f1_score
        elif metric_name == 'f1_weighted':
            return lambda labels, preds: f1_score(
                labels, preds, average='weighted'
            )
        elif metric_name == 'f1_micro':
            return lambda labels, preds: f1_score(
                labels, preds, average='micro'
            )
        elif metric_name == 'f1_macro':
            return lambda labels, preds: f1_score(
                labels, preds, average='macro'
            )

    def _next_i(self):
        i = self.results.shape[0]
        if i == 0:
            i = 0
        else:
            i += 1
        return i

    def _print_metric(self, average, deviation):
        avg = round(100 * average, 3)
        std = round(100 * deviation, 3)
        return "{} +/- {}".format(avg, std)

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

    def _select_folds(self, fold, cv):
        folds = np.arange(1, cv + 1)
        return folds[np.arange(folds.shape[0]) != fold - 1]
 
    def cross_evaluate_model_with_folds(self, cross_eval_tuple):
        """
        Evaluate with cross validation, using the pre defined folds of
        the dataset.
        """
        _model = cross_eval_tuple.classifier
        model = self.models[_model]
        _features = cross_eval_tuple.features
        features = self.features[_features]
        _labels = cross_eval_tuple.labels
        labels = self.labels[_labels]
        dataset = self.datasets[_labels]

        cv = dataset.dataframe['fold'].value_counts().shape[0]
        logger.debug('Cross validating: dataset={}, cv={}'.format(_labels,cv))

        metrics = np.zeros((cv, len(self.metrics_defs)))
        for i in range(1, cv + 1):
            train_folds = self._select_folds(i, cv)
            
            train_index = dataset.dataframe['fold'].isin(train_folds)
            train_index = np.array(train_index[train_index].index.tolist())
            test_index = dataset.dataframe['fold'].isin((i,))
            test_index = np.array(test_index[test_index].index.tolist())

            train_vecs = features.values[train_index]
            test_vecs = features.values[test_index]
            train_labels = labels[train_index].astype(int)
            test_labels = labels[test_index].astype(int)

            classifier = copy.deepcopy(model.classifier)
            classifier.fit(train_vecs, train_labels)
            predictions = classifier.predict(test_vecs)

            ms = np.zeros((len(self.metrics_defs), ))
            for m_i, metric in enumerate(self.metrics_defs):
                func = self._select_metric_call(metric)
                ms[m_i] = func(test_labels, predictions)
            metrics[i - 1, :] = ms

        average = np.average(metrics, axis=0)
        std = np.std(metrics, axis=0)
        scores = []
        for i in range(len(average)):
            scores.append(self._print_metric(average[i], std[i]))

        result = [_labels, _features, _model, 'static']
        result.extend(scores)
        i = self._next_i()
        self.results.loc[i, :] = result

    def cross_evaluate_model(self, cross_eval_tuple, multi_core):
        """
        Evaluate model with cross validation.
        """
        if 'fold' in self.datasets[cross_eval_tuple.labels].dataframe.columns:
            # If the dataset has pre-defined folds, use them
            folds_names =  self.datasets[cross_eval_tuple.labels].dataframe['fold'].unique()
            if not set(folds_names) == set(self.folds_defs):
                # The folds are not predefined (train, dev, test)
                self.cross_evaluate_model_with_folds(cross_eval_tuple)
                return

        _model = cross_eval_tuple.classifier
        model = self.models[_model]
        _features = cross_eval_tuple.features
        features = self.features[_features]
        _labels = cross_eval_tuple.labels
        labels = self.labels[_labels]
        folds = cross_eval_tuple.folds
        logger.debug(
            """Cross evaluating: model={}, features={}, folds={}, labels={}""".format(
                _model, _features, folds, _labels)
        )

        n_jobs = -1 if multi_core else 1
        logger.debug('Using n_jobs={}'.format(n_jobs))

        metrics = [None] * len(self.metrics_defs)
        for metric in cross_eval_tuple.metrics:
            args = [model.classifier, features.values, labels]
            kwargs = {'scoring': metric, 'cv': folds, 'n_jobs': n_jobs}
            try:
                value = cross_val_score(*args, **kwargs)
            except: # If multi processing is not supported
                logger.debug('Fallback to n_jobs=1')
                del kwargs['n_jobs']
                value = cross_val_score(*args, **kwargs)
                 
            avg, std = round(np.average(value), 4), round(np.std(value), 4)
            i = self.metrics_defs.index(metric)
            metrics[i] = self._print_metric(avg, std)

        result = [_labels, _features, _model, 'random']
        result.extend(metrics)

        i = self._next_i()
        self.results.loc[i, :] = result


    def evaluate(self, multi_core=True):
        """
        Evaluate all models on all datasets.
        If possible, will try to perform multi-core computation.
        """

        for tuple in self.tuples:
            if isinstance(tuple, EvalTuple):
                self.evaluate_model(tuple.features, tuple.classifier)
            elif isinstance(tuple, CrossEvalTuple):
                self.cross_evaluate_model(tuple, multi_core)
            else:
                raise ValueError('tuple is not correct')

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

            i = self._next_i()
                
            result = [test_dataset_name, features_name, model_name, False]
            result.extend(scores)

            self.results.loc[i, :] = result
