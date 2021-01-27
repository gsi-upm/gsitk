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
Processing of the PL04 dataset.

URL:
http://www.cs.cornell.edu/people/pabo/movie-review-data/
REF:
Bo Pang and Lillian Lee, 
A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts
Proceedings of ACL 2004. 
"""

import os
import logging
import pandas as pd
import numpy as np
from glob import glob
from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


fold_limits = zip(np.arange(0, 1000, 100), np.arange(99, 1000, 100))
folds = {i+1: limit  for i, limit in enumerate(fold_limits)}


class Pl04(Dataset):

    def _read_file(self, path, binary=False):
        with open(path, 'r') as f:
            content = f.read()
        return content

    def _get_file_cv_id(self, path):
        filename = os.path.splitext(os.path.basename(path))[0]
        id_ = filename.split('_')[-1]
        cv = filename.split('_')[0]
        cv = cv.replace('cv', '')
        return cv, id_
    
    def _choose_fold(self, cv, folds):
        cv = int(cv)
        for i, limits in folds.items():
            if cv >= limits[0] and cv <= limits[1]:
                return i

    def normalize_data(self):
        dataset = pd.DataFrame(columns=['id', 'fold', 'text', 'polarity'])
        raw_datapath = os.path.join(self.data_path,
                                     self.info['properties']['data_file'])
        logger.debug('Normalizing PL04')
        get_pol = lambda p: 1 if p == 'pos' else -1
        count = 0
        for pol in ('pos', 'neg'):
            for file in glob(os.path.join(raw_datapath, '{}/*'.format(pol))):
                text = self._read_file(file)
                cv, id_ = self._get_file_cv_id(file)
                fold = self._choose_fold(cv, folds)
                polarity = get_pol(pol)
                dataset.loc[count, :] = [id_, fold, text, polarity]
                count += 1
        normalized_text = normalize.normalize_text(dataset)
        dataset['text'] = normalized_text
        return dataset

