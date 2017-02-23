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
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


NAME = os.path.splitext(os.path.basename(__file__))[0]
INFO = utils.load_info(NAME)


def check_dataset():
    """Check the dataset, giving stats."""
    data_path = os.path.join(config.DATA_PATH, NAME)
    processed_path = os.path.join(data_path, INFO['properties']['processed_file'])

    return utils._check_datasets(processed_path, NAME)


def maybe_download():
    utils._maybe_download(NAME, INFO['properties']['url'], INFO['properties']['filename'],
                          INFO['properties']['expected_bytes'], INFO['properties']['sha256'])


def normalize_data(data_path):
    raw_data_path = os.path.join(data_path, INFO['properties']['data_file'])

    data = pd.read_csv(raw_data_path,
                       header=None,
                       encoding='latin-1',
                       index_col=False,
                       names = ['polarity', 'id', 'date', 'query', 'user', 'text'])

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


def prepare_data(download=True):
    """Prepare the data. All the steps are done if they have not been already stored in local.
        1. Download
        2. Extract from the original zip file
        3. Normalize the text
        4, Put in a custom pandas dataframe
    """
    logger.debug('Preparing data: {}'.format(NAME))

    if download:
        maybe_download()

    data_path = os.path.join(config.DATA_PATH, NAME)
    file_path = os.path.join(data_path, INFO['properties']['filename'])
    processed_path = os.path.join(data_path, INFO['properties']['processed_file'])


    if not os.path.exists(os.path.join(data_path, INFO['properties']['data_file'])):
        utils.extract_zip(file_path, data_path)

    if not os.path.exists(processed_path):
        logger.debug("Preprocessing {} data".format(NAME))
        normalized = normalize_data(data_path)

        logger.debug("Storing pre-processed data...")
        normalized.to_csv(processed_path)
        final_data = normalized

    else:
        final_data = pd.read_csv(processed_path)[['polarity', 'text']]

    logger.debug('{} data is ready'.format(NAME))
    return final_data
