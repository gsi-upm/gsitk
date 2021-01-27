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
Processing of the imdb dataset.

URL:
http://ai.stanford.edu/~amaas/data/sentiment/    
REF:
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). 
Learning word vectors for sentiment analysis. 
In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1 (pp. 142-150). Association for Computational Linguistics.
"""

import sys
import os
import logging
import codecs
import pandas as pd
import numpy as np
from glob import glob
from itertools import islice
from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


class Imdb(Dataset):

    def _extract_metadata(self, file):
        '''
        Gets the metadata (id, rating) from filename
        '''                    
        filename = os.path.splitext(os.path.basename(file))[0]
        id_, rating = filename.split('_')
        id_ = int(id_)
        return id_, rating

    def _say_progress(self, subset, count):
        '''
        Just give me some sense of progress.
        '''
        logger.info('At {} in {}'.format(count, subset))
        sys.stdout.flush()

    def populate_data(self, path, dataframe, relative_path=None, unsup=False, progress=10000, limit=None):
        '''
        Read the train, test or train/unsup directories and puts the data in the passed dataframe.
        '''
        pols = ('pos','neg')
        count = 0
        if unsup:
            for file in islice(glob(os.path.join(path, '*')), limit):
                with codecs.open(file, 'r', 'utf-8') as f:
                    text = f.read()
                    id_, rating = self._extract_metadata(file)
                    polarity = None
                    dataframe.loc[count, :] = [id_, text, polarity]
                    count += 1
                    if count % progress == 0:
                        self._say_progress('unsup', count)
            return dataframe
        
        for pol in pols:
            for file in islice(glob(os.path.join(path, '{}/{}/*'.format(relative_path, pol))),limit):
                with codecs.open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    id_, rating = self._extract_metadata(file)
                    polarity = None
                    fold = relative_path
                    if pol == 'pos':
                        polarity = 1
                    else:
                        polarity = -1
                    dataframe.loc[count, :] = [id_, fold, text, polarity, rating]
                    count += 1
                    if count % progress == 0:
                        self._say_progress(relative_path, count)       
        return dataframe

    def normalize_data(self):
        dataset_train = pd.DataFrame(columns=['id', 'fold', 'text', 'polarity', 'rating'])
        dataset_test = pd.DataFrame(columns=['id', 'fold', 'text', 'polarity', 'rating'])
        raw_datapath = os.path.join(self.data_path, self.info['properties']['data_file'])
        self.populate_data(raw_datapath, dataset_train, 'train')
        self.populate_data(raw_datapath, dataset_test, 'test')
        dataset = dataset_train.append(dataset_test, ignore_index=True)
        normalized_text = normalize.normalize_text(dataset)
        dataset['text'] = normalized_text
        
        return dataset


