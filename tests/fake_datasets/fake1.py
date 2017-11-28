"""
Processing of the Semeval2014 dataset.

URL:
http://alt.qcri.org/semeval2014/task9/

REF:
Rosenthal, S., Ritter, A., Nakov, P., & Stoyanov, V. (2014, August). Semeval-2014 task 9: Sentiment analysis in twitter. 
In Proceedings of the 8th international workshop on semantic evaluation (SemEval 2014) (pp. 73-80)."""

import os
import logging
import pandas as pd

from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


class Fake1(Dataset):

    def normalize_data(self):
        raw_data_path = os.path.join(self.data_path,
                                     self.info['properties']['data_file'])

        data = pd.read_csv(
            raw_data_path,
            encoding='utf-8',
            sep='\t',
            index_col=False,
        )

        if len(data) < 1:
            return data

        text_data = normalize.normalize_text(data)
        data = pd.concat([data['polarity'], text_data], axis=1)
        data.columns = ['polarity', 'text']
        # Remove text that was not fetched from source
        remove = lambda l: l != ['not', 'available']
        data = data.loc[data['text'].apply(remove)].reset_index(drop=True)

        return data
