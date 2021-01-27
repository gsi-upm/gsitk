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

from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


class Sentiment140(Dataset):

    def normalize_data(self):
        raw_datapath = os.path.join(self.data_path,
                                     self.info['properties']['data_file'])

        data = pd.read_csv(
            raw_datapath,
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
