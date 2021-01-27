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
Processing of the Semeval2014 dataset.

URL:
http://alt.qcri.org/semeval2014/task9/

REF:
Rosenthal, S., Ritter, A., Nakov, P., & Stoyanov, V. (2014, August). Semeval-2014 task 9: Sentiment analysis in twitter. 
In Proceedings of the 8th international workshop on semantic evaluation (SemEval 2014) (pp. 73-80)."""

import os
import logging
import re
import pandas as pd

from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


class Semeval14(Dataset):

    def normalize_data(self):
        raw_datapath = os.path.join(self.data_path,
                                     self.info['properties']['data_file'])

        data = pd.read_csv(
            raw_datapath,
            header=None,
            encoding='utf-8',
            sep='\t',
            index_col=False,
            names = [
                'tweet_id',
                'user_id',
                'polarity',
                'text',
                
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
        # Remove text that was not fetched from source
        remove = lambda l: l != ['not', 'available']
        data = data.loc[data['text'].apply(remove)].reset_index(drop=True)

        return data
