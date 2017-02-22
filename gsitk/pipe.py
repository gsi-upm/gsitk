"""
Pipelining the different parts of the module.
"""

from collections import namedtuple

# Dataset = namedtuple('Dataset', ['name', 'values'])
Features = namedtuple('Features', ['name', 'dataset', 'values'])
Model = namedtuple('Model', ['name', 'classifier'])
EvalTuple = namedtuple('EvalTuple', ['classifier', 'features', 'labels'])
