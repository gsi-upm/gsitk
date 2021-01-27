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
Processing of the imdb_unsup dataset.

URL:
http://ai.stanford.edu/~amaas/data/sentiment/    
REF:
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). 
Learning word vectors for sentiment analysis. 
In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1 (pp. 142-150). Association for Computational Linguistics.
"""

import os
import logging
import pandas as pd
import numpy as np
from glob import glob
from itertools import islice
from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize
from gsitk.datasets.imdb import Imdb

logger = logging.getLogger(__name__)


class Imdb_unsup(Imdb):

    def normalize_data(self):
        dataset = pd.DataFrame(columns=['id', 'text', 'polarity'])
        data_path = os.path.join(self.data_path, self.info['properties']['data_file'])
        self.populate_data(path=data_path, dataframe=dataset, unsup=True)
        return dataset

