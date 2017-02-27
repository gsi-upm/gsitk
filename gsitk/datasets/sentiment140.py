"""
Processing of the sentiment140 dataset.

URL:
http://help.sentiment140.com/for-students/

REF:
Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision.
CS224N Project Report, Stanford, 1(12).
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


class Sentiment140(Dataset):

    def __init__(self, info=None):
        if info is None:
            info = utils.load_info(NAME) 
        super(Sentiment140, self).__init__(info)

    def normalize_data(self):
        data_path = os.path.join(config.DATA_PATH, self.name)
        raw_data_path = os.path.join(data_path,
                                     self.info['properties']['data_file'])

        data = pd.read_csv(
            raw_data_path,
            header=None,
            encoding='latin-1',
            index_col=False,
            names = [
                'polarity',
                'id',
                'date',
                'query',
                'user',
                'text'
            ]
        )

        # Convert the raw polarity values to a [-1,1] range
        pol_conv = {
            0: -1,
            2: 0,
            4: 1
        }

        data['polarity'].replace(pol_conv, inplace=True)

        # Tokenize and clean the test
        text_data = normalize.normalize_text(data)
        data = pd.concat([data['polarity'], text_data], axis=1)

        data.columns = ['polarity', 'text']

        return data
