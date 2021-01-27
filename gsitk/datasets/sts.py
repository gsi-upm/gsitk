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
Processing of the STS dataset.

URL:
Forked from: https://github.com/pollockj/world_mood
https://github.com/oaraque/world_mood/tree/master/sts_gold_v03
REF:
Saif, H., Fernandez, M., He, Y., Alani, H.
Evaluation datasets for twitter sentiment analysis.
Proceedings, 1st Workshop on Emotion and Sentiment in Social and Expressive Media (ESSEM) in conjunction with AI*IA Conference.
Turin, Italy (2013)
"""

import os
import logging
import pandas as pd

from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


class STS(Dataset):

    def normalize_data(self):
        raw_datapath = os.path.join(self.data_path,
                                     self.info['properties']['data_file'])
        data = pd.read_csv(raw_datapath, sep=';')

        data = data[['polarity', 'tweet']]
        data['polarity'] = data['polarity'].apply(lambda p: 1 if p == 4 else -1)
        data['text'] = data['tweet']

        # Tokenize and clean the test
        text_data = normalize.normalize_text(data)
        data = pd.concat([data['polarity'], text_data], axis=1)

        data.columns = ['polarity', 'text']

        return data
