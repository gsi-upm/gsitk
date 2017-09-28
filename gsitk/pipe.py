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

def EvalPipeline(pipeline, dataset, name=None, features=None):
    name_ = name
    if name is None:
        name_ = '{}__{}'.format(pipeline.name, dataset)

    return _EvalPipeline(name_, pipeline, dataset, features)
