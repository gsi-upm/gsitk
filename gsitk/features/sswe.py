"""
Feature extraction with SSWE, as explained in

"Enhancing deep learning sentiment analysis with ensemble techniques in social
applications",
http://dx.doi.org/10.1016/j.eswa.2017.02.002


Needs a SSWE model previously trained.
The original model is described in
Tang, D., Wei, F., Yang, N., Zhou, M., Liu, T., & Qin, B. (2014, June).
Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification.
In ACL (1) (pp. 1555-1565).
"""

import os
import yaml
import logging
import csv
import numpy as np

from gsitk.config import default
from gsitk.datasets import utils as dataset_utils
from gsitk.features.embeddings import Embedding

from sklearn.base import TransformerMixin

config = default()

logger = logging.getLogger(__name__)


class SSWE(Embedding, TransformerMixin):
    def __init__(self, convolution=[1, 0, 0], download=True):
        super(SSWE, self).__init__(convolution)
        self.info = self._load_info()
        self._download = download
        self._download_model()
        self.model = self.load_sswe()
        self.size = self._get_size()

    def _load_info(self):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(path, 'sswe.yml'), 'r') as f:
            info = yaml.load(f)
        return info

    def _download_model(self):
        if self._download:
            dataset_utils._maybe_download(self.info['name'],
                                      self.info['url'],
                                      self.info['filename'],
                                      self.info['expected_bytes'],
                                      self.info['sha256'])

    def load_sswe(self, path=config.DATA_PATH):
        data_path = os.path.join(path,
                                 self.info['name'])
        file_path = os.path.join(data_path, self.info['data_file'])
        zip_path = os.path.join(data_path, self.info['filename'])

        if not os.path.exists(file_path):
            logger.debug('Extracting {}...'.format(zip_path))
            dataset_utils.extract_zip(zip_path, data_path)

        sswe = dict()
        with open(file_path) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in reader:
                word = line[0]
                vec = np.array(line[1:], dtype='float64')
                sswe[word] = vec
        return sswe

    def _get_size(self):
        """Supposing all vectors are equal. They should be."""
        for vector in self.model.values():
            size = vector.shape[0]
        return size

    def transform(self, X):
        """Extract the features from normalized text.
        This considers X to be a list of lists of texts.
        [
        ['my', 'dog', 'run', 'in', 'the', 'rain']
        ]

        w2v_format can be 'gensim', 'google_txt' or 'google_bin'
        """
        vecs = self.comments2vec(text=X)

        vecs = self.check_vector(vecs) 

        return vecs

    def fit(self, x, y=None):
        return self
