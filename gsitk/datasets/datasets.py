"""
Access operations to the available datasets.
"""

import os
import glob
import importlib
from gsitk import config
from gsitk.datasets import utils


def view_datasets(pprint=True):
    """Check the available datasets."""
    path = os.path.dirname(os.path.abspath(__file__))
    datasets = [dataset for dataset in glob.glob(os.path.join(path, '*.yaml'))]
    datasets.extend([dataset for dataset in glob.glob(os.path.join(path, '*.yml'))])

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


def get_dataset_names():
    """Get all the available dataset names."""
    path = os.path.dirname(os.path.abspath(__file__))
    datasets = [dataset for dataset in glob.glob(os.path.join(path, '*.yaml'))]
    datasets.extend([dataset for dataset in glob.glob(os.path.join(path, '*.yml'))])

    all = []
    for dataset in datasets:
        info = utils.load_info(dataset, given_path=True)
        name = info['properties']['name']
        all.append(name)
    return all


def prepare_datasets(datasets=[]):
    """Prepare all the specified datasets. If datasets is None, prepare all."""
    data = dict()

    if len(datasets) == 0:
        names = get_dataset_names()
    else:
        names = datasets

    for name in names:
        dataset_module = importlib.import_module('gsitk.datasets.{}'.format(name))
        prepared = dataset_module.prepare_data()
        data[name] = prepared

    return data