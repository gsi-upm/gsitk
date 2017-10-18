"""
Evaluation of classifiers on datasets.
"""

import logging
from collections import namedtuple
import itertools
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import cross_val_score

from gsitk.pipe import EvalTuple, CrossEvalTuple, _EvalPipeline, EvalPipeline

logger = logging.getLogger(__name__)


class Prediction():
    def __init__(self, name, dataset, features, model, values, description):
        self.name = name
        self.dataset = dataset
        self.features = features
        self.model = model
        self.values = values
        self.description = description


class Evaluation():
    """
    Models an evaluation process.
    """
    def __init__(self, tuples=None, datasets=None, pipelines=None, features=None):
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

        self.metrics_defs = ['accuracy', 'precision_macro', 'recall_macro',
                             'f1_weighted', 'f1_micro', 'f1_macro']

        self.folds_defs = ['train', 'dev', 'test']

        #self.models = self._models_setup(models)
        self.features = self._features_setup(features)
        self.datasets = self._datasets_setup(datasets)
        self.pipelines = self._pipelines_setup(pipelines) if not pipelines is None else pipelines
        self.tuples = tuples if not tuples is None else self.generate_tuples()
        self.predictions = dict()
        self.labels = self._labels_setup()
        self.results = self._results_setup()
        self.descriptions = dict()
        
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
        if features is None:
            return None
        _features = dict()
        for features_ in features:
            _features[features_.name] = features_
        return _features

    def _labels_setup(self):
        _labels = dict()
        for dataset_name, dataset_obj in self.datasets.items():
            _labels[dataset_name] = dataset_obj.dataframe['polarity'].values.astype(int)
        return _labels

    def _pipelines_setup(self, pipelines):
        _pipelines = dict()
        for pipeline in pipelines:
            assert isinstance(pipeline.name, str)
            _pipelines[pipeline.name] = pipeline
        return _pipelines

    def _select_metric_call(self, metric_name):
        if metric_name == 'accuracy':
            return accuracy_score
        elif metric_name == 'recall':
            return recall_score
        elif metric_name == 'recall_macro':
            return lambda labels, preds: recall_score(
                labels, preds, average='macro'
            )
        elif metric_name == 'precision':
            return precision_score
        elif metric_name == 'precision_macro':
            return lambda labels, preds: precision_score(
                labels, preds, average='macro'
            )
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
        return self.results.shape[0]
        
    def _print_metric(self, average, deviation):
        avg = round(100 * average, 3)
        std = round(100 * deviation, 3)
        return "{} +/- {}".format(avg, std)

    def _generate_pipeline_description(self, pipeline):
        steps = []
        for step in pipeline.steps:
            name = None
            try:
                name = pipeline.named_steps[step[0]].name
            except AttributeError:
                # It does not have a name
                pass

            if not name is None:
                steps.append('{}({})'.format(step[0], name))
            else:
                steps.append('{}'.format(step[0]))

        return ' --> '.join(steps)

    def _add_metadata(self):
        for i, desc in self.descriptions.items():
            self.results.loc[i, 'Description'] = desc

    def generate_tuples(self):
        tuples = []
        for combination in itertools.product(self.datasets.keys(), self.pipelines.keys()):
            tuples.append(
                EvalPipeline(self.pipelines[combination[1]], combination[0])
            )
        return tuples

    def evaluate_model(self, features_name, model):
        """Predict from features."""
        logger.debug('Model {} predicting from features {}'.format(model.name, features_name))
        _features = self.features[features_name]
        #_model = self.models[model_name]
        predictions = model.predict(_features.values)
        prediction = self.Prediction('{}__{}'.format(model.name, features_name),
                                     _features.dataset,
                                     _features.name,
                                     model.name,
                                     predictions)
        self.predictions['{}__{}'.format(model.name, features_name)] = prediction

    def evaluate_pipeline(self, tuple):
        """Predict from dataset using pipeline"""
        logger.debug('Pipeline {} predicting from dataset {}'.format(
            tuple.pipeline.named_steps, tuple.dataset
        ))
        if not tuple.features is None:
            _input = self.features[tuple.features].values
            input_name = self.features[tuple.features].name
        else: # No features
            if 'fold' in self.datasets[tuple.dataset].dataframe.columns:
                # If the dataset has pre-defined folds, use them
                test_indices = (self.datasets[tuple.dataset].dataframe['fold'] == 'test').values
                _input = self.datasets[tuple.dataset].dataframe['text'].values[test_indices]
            else:
                _input = self.datasets[tuple.dataset].dataframe['text'].values

            input_name = tuple.dataset

        predictions = tuple.pipeline.predict(_input)
        description = self._generate_pipeline_description(tuple.pipeline)
        prediction = Prediction('{}__{}'.format(tuple.name, input_name),
                                     tuple.dataset,
                                     None,
                                     tuple.name,
                                     predictions,
                                     description)
        self.predictions['{}__{}'.format(tuple.name, input_name)] = prediction
        
    def _select_folds(self, fold, cv):
        folds = np.arange(1, cv + 1)
        return list(folds[np.arange(folds.shape[0]) != fold - 1])
 
    def cross_evaluate_model_with_folds(self, cross_eval_tuple):
        """
        Evaluate with cross validation, using the pre defined folds of
        the dataset.
        """
        model = cross_eval_tuple.classifier
        _features = cross_eval_tuple.features
        features = self.features[_features]
        _labels = cross_eval_tuple.labels
        labels = self.labels[_labels]
        dataset = self.datasets[_labels]

        cv = dataset.dataframe['fold'].value_counts().shape[0]
        logger.debug('Cross validating: dataset={}, cv={}, feats={}'.format(_labels,cv, _features))

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

            classifier = copy.deepcopy(model)
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

        result = [_labels, _features, model.name, 'static']
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

        model = cross_eval_tuple.classifier
        _features = cross_eval_tuple.features
        features = self.features[_features]
        _labels = cross_eval_tuple.labels
        labels = self.labels[_labels]
        folds = cross_eval_tuple.folds
        logger.debug(
            """Cross evaluating: model={}, features={}, folds={}, labels={}""".format(
                model, _features, folds, _labels)
        )

        n_jobs = -1 if multi_core else 1
        logger.debug('Using n_jobs={}'.format(n_jobs))

        metrics = [None] * len(self.metrics_defs)
        for metric in cross_eval_tuple.metrics:
            args = [model, features.values, labels]
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

        result = [_labels, _features, model.name, 'random']
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
            elif isinstance(tuple,  _EvalPipeline):
                self.evaluate_pipeline(tuple)
            else:
                raise ValueError('tuple is not correct')

        for prediction in self.predictions.values():
            labels = None
            if 'fold' in self.datasets[prediction.dataset].dataframe.columns:
                # If the dataset has pre-defined folds, use them
                test_indices = (self.datasets[prediction.dataset].dataframe['fold'] == 'test').values
                labels = self.labels[prediction.dataset][test_indices]
            else:
                labels = self.labels[prediction.dataset]
            preds = prediction.values
            test_dataset_name = prediction.dataset
            features_name = prediction.features
            model_name = prediction.model

            scores = list()
            for metric in self.metrics_defs:
                func = self._select_metric_call(metric)
                logger.debug('evaluating on metric {}'.format(func))
                scores.append(func(labels, preds))

            i = self._next_i()

            result = [test_dataset_name, features_name, model_name, False]
            result.extend(scores)

            self.results.loc[i, :] = result

            try:
                description = prediction.description
                self.descriptions[i] = description
            except AttributeError:
                # If prediction has no description, do not add to results table
                pass

        self._add_metadata()
