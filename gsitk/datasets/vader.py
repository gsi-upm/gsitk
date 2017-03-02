"""
Processing of the vader dataset.

URL:
https://github.com/cjhutto/vaderSentiment
REF:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""

import os
import logging
import pandas as pd

from gsitk import config
from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


NAME = os.path.splitext(os.path.basename(__file__))[0]


class Vader(Dataset):

    def __init__(self, info=None):
        if info is None:
            info = utils.load_info(NAME)
        super(Vader, self).__init__(info)

    def _labelize(self,polarity):
        polarity = float(polarity)
        if polarity >= 0:
            return 1
        else:
            return -1

    def normalize_data(self):
        data_path = os.path.join(config.DATA_PATH, self.name)
        raw_data_path = os.path.join(data_path,
                                     self.info['properties']['data_file'])

        data = pd.read_csv(
            raw_data_path,
            header=None,
            index_col=False,
            sep='\t',
            names = [
                'id',
                'polarity',
                'text'
            ]
        )

        # Round polarity value
        data['polarity'] = data['polarity'].apply(self._labelize)
        # Tokenize and clean the test
        text_data = normalize.normalize_text(data)
        data = pd.concat([data['polarity'], text_data], axis=1)

        data.columns = ['polarity', 'text']

        return data
