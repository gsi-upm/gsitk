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
