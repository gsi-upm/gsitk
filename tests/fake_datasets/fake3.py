"""
Fake dataset for tests.
"""

import os
import logging
import pandas as pd

from gsitk.datasets.datasets import Dataset

logger = logging.getLogger(__name__)


class Fake3(Dataset):
    is_dataset = True

    def normalize_data(self):
        data = pd.DataFrame([['so sad', -1], ['so happy', 1]])

        data.columns = ['polarity', 'text']

        return data
