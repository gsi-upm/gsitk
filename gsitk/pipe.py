"""
Pipelining the different parts of the module.
"""

from collections import namedtuple


Features = namedtuple('Features', ['name', 'dataset', 'values'])

Model = namedtuple('Model', ['name', 'classifier'])

EvalTuple = namedtuple('EvalTuple', ['classifier', 'features', 'labels'])

CrossEvalTuple = namedtuple('CrossEvalTuple',
                            [
                                'classifier',
                                'features',
                                'labels',
                                'metrics',
                                'folds'
                            ])

_EvalPipeline = namedtuple('EvalPipeline', ['name', 'pipeline', 'dataset', 'features'])

def EvalPipeline(name, pipeline, dataset, features=None):
    return _EvalPipeline(name, pipeline, dataset, features)
