"""
Processing of the SemEval2007 dataset, Affective Text.

URL:
http://nlp.cs.swarthmore.edu/semeval/tasks/task14/summary.shtml

REF:
Strapparava, C., & Mihalcea, R. (2007, June). Semeval-2007 task 14: Affective text.
In Proceedings of the 4th International Workshop on Semantic Evaluations (pp. 70-74).
Association for Computational Linguistics.
"""

import os
import logging
import pandas as pd
from bs4 import BeautifulSoup

from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


class Semeval07(Dataset):

    def _read_xml_file(self, file_path):
        with open(file_path, 'r') as f:
            file_data = f.read()

        id_text = list()
        soup = BeautifulSoup(file_data, 'lxml')
        for instance in soup.find_all('instance'):
            id_text.append([int(instance.get('id')), instance.get_text()])
        return pd.DataFrame(id_text, columns=['id', 'text']).set_index('id')

    def _read_emo_annotation(self, file_path):
        annot = pd.read_csv(file_path, sep=' ', header=None,
                            names=['id',  'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'])
        annot = annot.set_index('id')
        annot = annot / 100 # normalize rage in [0, 1]
        return annot

    def _read_valence_annotation(self, file_path):
        annot = pd.read_csv(file_path, sep=' ', header=None,
                            names=['id',  'valence'])
        annot = annot.set_index('id')
        annot = annot / 100
        return annot

    def normalize_data(self):
        
        dev_text = self._read_xml_file(os.path.join(self.data_path, 'trial/affectivetext_trial.xml'))
        dev_valence = self._read_valence_annotation(os.path.join(self.data_path, 'trial/affectivetext_trial.valence.gold'))
        dev_emotion = self._read_emo_annotation(os.path.join(self.data_path, 'trial/affectivetext_trial.emotions.gold'))
        dev = pd.concat([dev_text, dev_valence, dev_emotion], axis=1)
        dev['fold'] = 'dev'

        test_text = self._read_xml_file(os.path.join(self.data_path, 'test/affectivetext_test.xml'))
        test_valence = self._read_valence_annotation(os.path.join(self.data_path, 'key/affectivetext_test.valence.gold'))
        test_emotion = self._read_emo_annotation(os.path.join(self.data_path, 'key/affectivetext_test.emotions.gold'))
        test = pd.concat([test_text, test_valence, test_emotion], axis=1)
        test['fold'] = 'test'

        data = pd.concat([dev, test], axis=0)

        # Tokenize and clean the test
        data['original_text'] = data['text'].copy()
        text_data = normalize.normalize_text(data)
        data['text'] = text_data

        return data
