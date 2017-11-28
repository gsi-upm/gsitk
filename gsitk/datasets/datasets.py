"""
Access operations to the available datasets.
"""

import os
import inspect
import logging
import yaml
import glob
import hashlib
import zipfile
from six.moves import urllib
import pandas as pd

from gsitk.config import default
from gsitk.datasets import utils

logger = logging.getLogger(__name__)


class Dataset():
    def __init__(self, info=None, config=None):
        """Inheritor class must assign these values."""
        if info is None:
            info = utils.load_info(self.__class__.__name__.lower())
        self.info = info
        self.name = self.info['properties']['name']
        if not config:
            config = default()
            config.DATA_PATH = os.path.join(config.DATA_PATH,
                                            self.name)
        self.config = config
        self.data_path = self.config.DATA_PATH

    @property
    def data(self):
        if not hasattr(self, '_data') or not self._data:
            self._data = self.prepare_data(download=self.info['properties'].get('url', None))
        return self._data

    def maybe_download(self, move=False):
        utils._maybe_download(data_name=self.name,
                              url=self.info['properties']['url'],
                              filename=self.info['properties']['filename'],
                              expected_bytes=self.info['properties']['expected_bytes'],
                              sha256=self.info['properties']['sha256'],
                              move=move)

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
        processed_path = os.path.join(self.data_path, self.info['properties']['processed_file'])

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
            if self.info['properties'].get('download') is None or \
               self.info['properties'].get('download') is True:

                if self.info['properties'].get('copy') is None or \
                   self.info['properties'].get('copy') is True:
                    # In case the dataset is batteries included, just copy it
                    self.maybe_download(move=True)
                else:
                    # Download data from original repository
                    self.maybe_download()
            else:
                logger.debug('Skipping download of {}'.format(self.name))

            
        data_file = os.path.join(self.data_path, self.info['properties']['data_file'])
        processed_path = os.path.join(self.data_path, self.info['properties']['processed_file'])
        
        if not os.path.exists(data_file):
            file_path = os.path.join(self.data_path, self.info['properties']['filename'])
            extract_func = self.extract_function()
            if not extract_func is None:
                extract_func(file_path, self.data_path)
            
        if not os.path.exists(processed_path):
            logger.debug("Preprocessing {} data".format(self.name))
            normalized = self.normalize_data()
            
            logger.debug("Storing pre-processed data...")
            normalized.to_pickle(processed_path)
            final_data = normalized
            
        else:
            final_data = pd.read_pickle(processed_path)

        #assert 'polarity' in final_data.columns Nope Nope
        assert 'text' in final_data.columns

        # Labels must be int values
        #try:
        #    final_data['polarity'] = final_data['polarity'].values.astype(int)
        #except TypeError as e:
        #    if final_data['polarity'].value_counts().shape == (0,):
        #        pass
        #    else:
        #        raise e

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
            stored_path = os.path.join(self.config.DATA_PATH, name, processed_name)
            response.append(utils._check_dataset(stored_path, name, count=count))

        if pprint:
            print(''.join(response))
        else:
            return response

    def find_datasets(self, path=None):
        path = path or os.path.dirname(os.path.abspath(__file__))
        extensions = ['yml', 'yml']
        info_files = []
        for ext in extensions:
            info_files.extend([dataset for dataset in \
                               glob.glob(os.path.join(path,
                                                      '*.{}'.format(ext)))])
        return info_files

    def get_dataset(self, info, root=None, data_path=None):

        if not isinstance(info, dict):
            root = root or os.path.dirname(os.path.abspath(info))
            info = utils.load_info(info, given_path=True)
        name = info['properties']['name']
        module_name = info['properties'].get('module',  name)
        dataset_module = utils.load_module(module_name, root=root)
        obj = None
        logger.info(inspect.getmembers(dataset_module))
        config = None
        if data_path:
            config = default()
            config.DATA_PATH = root
        for data_name, data_class in inspect.getmembers(dataset_module):
            if inspect.isclass(data_class) and \
               data_name.lower() == module_name.lower():
                obj = data_class(info, config)
                break
        if not obj:
            raise ImportError(('Module {} not found '
                               'the desired class.').format(name))
        return obj

    def get_datasets(self):
        """Get all the available dataset names."""
        info_files = self.find_datasets()
        infos = dict()
        objs = dict()
        for info in info_files:
            dataset = self.get_dataset(info)
            info = dataset.info
            infos[info['properties']['name']] = info
            objs[info['properties']['name']] = dataset

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

