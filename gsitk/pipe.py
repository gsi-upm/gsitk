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
