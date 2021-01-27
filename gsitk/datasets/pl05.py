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
Processing of the PL05 dataset.

URL:
http://www.cs.cornell.edu/people/pabo/movie-review-data/
REF:

Bo Pang and Lillian Lee
Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales
Proceedings of ACL 2005.
"""

import os
import logging
import codecs
import pandas as pd
import numpy as np
from glob import glob
from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


class Pl05(Dataset):

    def normalize_data(self):
        dataset = pd.DataFrame(columns=['id', 'text', 'polarity'])
        self.data_path = os.path.join(self.data_path, self.info['properties']['data_file'])

        logger.debug('Normalizing PL05')

        get_pol = lambda p: 1 if p == 'pos' else -1

        texts = list()
        polarities = list()
        for f in glob(os.path.join(self.data_path, '*')):
            polarity = get_pol(f.split('.')[-1])
            with codecs.open(f, 'r', encoding='ISO-8859-2') as f_:
                lines = f_.readlines()
            texts.append(lines)
            polarity_column = (polarity * np.ones(len(lines))).astype(int)
            polarities.append(polarity_column)
        
        dataset['polarity'] = np.concatenate(polarities).astype(int)
        dataset['text'] = np.concatenate(texts)
        dataset['id'] = np.arange(dataset.shape[0])

        normalized_text = normalize.normalize_text(dataset)
        dataset['text'] = normalized_text
        return dataset
