"""
Access operations to the available datasets.
"""

import os
import inspect
import logging
import yaml
import glob
import importlib
import hashlib
import zipfile
from six.moves import urllib
import pandas as pd

from gsitk import config
from gsitk.datasets import utils

logger = logging.getLogger(__name__)


class Dataset():
    def __init__(self, info):
        """Inheritor class must assign these values."""
        self.info = info
        self.name = self.info['properties']['name']

    def maybe_download(self):
        utils._maybe_download(data_name=self.name,
                              url=self.info['properties']['url'],
                              filename=self.info['properties']['filename'],
                              expected_bytes=self.info['properties']['expected_bytes'],
                              sha256=self.info['properties']['sha256'])

    def extract_function(self):
        """
        Select appropiate method of decompression.
        """
        if self.info['properties']['compression'] is None:
            return None

        type_ = self.info['properties']['compression']['type']
        if type_ == 'zip':
            return utils.extract_zip
        elif type_ == 'targz':
            return utils.extract_targz
        else:
            return None

    def _check_dataset(self, path, name, count=None):

        nlines = 0
        if os.path.exists(path):
            downloaded = True
            if count is None:
                nlines = utils.file_len(path)
            else:
                nlines = count
        else:
            downloaded = False

        response = """- {}:
        \t Downloaded: {}
        \t # instances: {}\n\n""".format(name, downloaded, nlines)

        return response

    def check_dataset(self):
        """Check the dataset, giving stats."""
        data_path = os.path.join(config.DATA_PATH, self.name)
        processed_path = os.path.join(data_path, self.info['properties']['processed_file'])

        return utils._check_dataset(processed_path, self.name)


    def prepare_data(self, download=True):
        """Prepare the data. All the steps are done if they have not been
        already stored in local.
        1. Download
        2. Extract from the original zip file
        3. Normalize the text
        4, Put in a custom pandas dataframe
        """
        logger.debug('Preparing data: {}'.format(self.name))
        
        if download:
            self.maybe_download()
            
        data_path = os.path.join(config.DATA_PATH, self.name)
        file_path = os.path.join(data_path, self.info['properties']['filename'])
        processed_path = os.path.join(data_path, self.info['properties']['processed_file'])
        
        
        if not os.path.exists(os.path.join(data_path, self.info['properties']['data_file'])):
            extract_func = self.extract_function()
            if not extract_func is None:
                extract_func(file_path, data_path)
            
        if not os.path.exists(processed_path):
            logger.debug("Preprocessing {} data".format(self.name))
            normalized = self.normalize_data()
            
            logger.debug("Storing pre-processed data...")
            normalized.to_pickle(processed_path)
            final_data = normalized
            
        else:
            final_data = pd.read_pickle(processed_path)[['polarity', 'text']]
            
        logger.debug('{} data is ready'.format(self.name))
        return final_data

    def normalize_data(self):
        """To be implemented by the inheritor class."""
        pass


class DatasetManager():
    def __init__(self):
        self.infos, self.datasets = self.get_datasets()

    def view_datasets(self, pprint=True):
        """Check the available datasets."""
        path = os.path.dirname(os.path.abspath(__file__))
        datasets = [dataset for dataset in \
                    glob.glob(os.path.join(path, '*.yaml'))]
        datasets.extend([dataset for dataset in \
                         glob.glob(os.path.join(path, '*.yml'))])

        response = []
        for dataset in datasets:
            info = utils.load_info(dataset, given_path=True)
            name = info['properties']['name']
            processed_name = info['properties']['processed_file']
            count = info['stats'].get('instances', None)
            stored_path = os.path.join(config.DATA_PATH, name, processed_name)
            response.append(utils._check_dataset(stored_path, name, count=count))

        if pprint:
            print(''.join(response))
        else:
            return response


    def get_datasets(self):
        """Get all the available dataset names."""
        path = os.path.dirname(os.path.abspath(__file__))
        info_files = [dataset for dataset in \
                    glob.glob(os.path.join(path, '*.yaml'))]
        info_files.extend([dataset for dataset in \
                         glob.glob(os.path.join(path, '*.yml'))])

        infos = dict() 
        objs = dict()
        for info_name in info_files:
            info = utils.load_info(info_name, given_path=True)
            name = info['properties']['name']
            infos[name] = info

            dataset_module = importlib.import_module(
                'gsitk.datasets.{}'.format(name)
            )
            found = False
            logger.info(inspect.getmembers(dataset_module))
            for data_name, data_class in inspect.getmembers(dataset_module):
                if inspect.isclass(data_class) and \
                   data_name.lower() == name.lower():
                    found = True
                    obj = data_class(info)
                    objs[name] = obj
                    break
            if not found:
                raise ImportError(('Module gsitk.datasets.{} does not contain '
                                   'the desired class.').format(name))

        return infos, objs

    def prepare_datasets(self, datasets=None, download=True):
        """Prepare all the specified datasets.
        If datasets is None, prepare all."""
        data = dict()

        if not datasets:
            names = list(self.infos.keys())
        else:
            names = datasets
        
        for name in names:
            prepared = self.datasets[name].prepare_data(download=download)
            data[name] = prepared

        return data

