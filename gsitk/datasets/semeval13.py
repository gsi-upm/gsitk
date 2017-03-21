"""
Processing of the semeval2013 dataset.

URL:
https://www.cs.york.ac.uk/semeval-2013/accepted/101_Paper.pdf

REF:
HLTCOE, J. (2013). SemEval-2013 Task 2: Sentiment Analysis in Twitter. Atlanta, Georgia, USA, 312.
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


class Semeval13(Dataset):

    def __init__(self, info=None):
        if info is None:
            info = utils.load_info(NAME) 
        super(Semeval13, self).__init__(info)

    def normalize_data(self):
        data_path = os.path.join(config.DATA_PATH, self.name)
        raw_data_path = os.path.join(data_path,
                                     self.info['properties']['data_file'])

        data = pd.read_csv(
            raw_data_path,
            header=None,
            encoding='utf-8',
            sep='\t',
            index_col=False,
            names = [
                'tweet_id',
                'user_id',
                'polarity',
                'text'
            ]
        )

        # Convert the raw polarity values to a [-1,1] range
        pol_conv = {
            "negative": -1,
            "neutral": 0,
            "positive": 1
        }

        data['polarity'].replace(pol_conv, inplace=True)
        # Tokenize and clean the test
        text_data = normalize.normalize_text(data)
        data = pd.concat([data['polarity'], text_data], axis=1)

        data.columns = ['polarity', 'text']

        return data