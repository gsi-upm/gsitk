"""
Utils for the datasets submodule.
"""


import os
import logging
import hashlib
import zipfile
import tarfile
import yaml
import shutil
import sys
import importlib
from six.moves import urllib


logger = logging.getLogger(__name__)


def load_info(name, given_path=False):
    """Load YAML info for dataset."""
    if not given_path:
        path = os.path.dirname(os.path.abspath(__file__))

        if os.path.exists(os.path.join(path, '{}.yml'.format(name))):
            ext = '.yml'
        elif os.path.exists(os.path.join(path, '{}.yaml'.format(name))):
            ext = '.yaml'
        else:
            raise FileNotFoundError("{}".format(name))

        with open(os.path.join(path, '{}{}'.format(name, ext)), 'r') as f:
            info = yaml.load(f)

    else:
        with open(name, 'r') as f:
            info = yaml.load(f)

    return info


def _maybe_download(data_name, url, filename, expected_bytes, sha256=None, move=False):
    """Download the dataset if it is not already stored locally.

    :return:
    """
    data_path = os.path.join(self.config.DATA_PATH, data_name)
    file_path = os.path.join(data_path, filename)

    logger.debug("Checking data path: {}".format(data_path))
    if not os.path.exists(file_path):
        logger.debug("Saving data in {}".format(data_path))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not move:
            logger.debug("Downloading... {}".format(filename))
            filename, _ = urllib.request.urlretrieve(url, file_path)
            logger.debug("Downloaded {}".format(filename))
        else:
            logger.debug("Moving {} to {}".format(filename, data_path))
            shutil.copy(os.path.join(self.config.RESOURCES_PATH, filename),
                        data_path)


    statinfo = os.stat(file_path)

    if statinfo.st_size == expected_bytes:
        if sha256 is not None:
            hash_f = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
            if hash_f == sha256:
                logger.debug("Verified: {}".format(filename))
            else:
                raise ValueError('hash of {} does not match original'.format(filename))
    else:
        raise Exception('Failed to verify {}'.format(filename))


def extract_zip(file_path, data_path):
    """Extract a zip file into a given path.

    :param file_path: zip file to extract
    :param data_path: path where to extract the zip
    """
    if not os.path.exists(file_path):
        raise ValueError("File to extract does not exist")

    if not os.path.exists(data_path):
        raise ValueError("Extraction path does not exist")

    with zipfile.ZipFile(file_path) as zip_:
        zip_.extractall(path=data_path)


def extract_targz(file_path, data_path):
    """Extract a tar file into a given path.

    :param file_path: tar file to extract
    :param data_path: path where to extract the tar
    """
    if not os.path.exists(file_path):
        raise ValueError("File to extract does not exist")

    if not os.path.exists(data_path):
        raise ValueError("Extraction path does not exist")

    with tarfile.open(file_path,'r:gz') as tar:
        tar.extractall(path=data_path)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i  # do not count to header line


def _check_dataset(path, name, count=None):

    nlines = 0
    if os.path.exists(path):
        downloaded = True
        if count is None:
            nlines = file_len(path)
        else:
            nlines = count
    else:
        downloaded = False

    response = """- {}:
    \t Downloaded: {}
    \t # instances: {}\n\n""".format(name, downloaded, nlines)

    return response


def load_module(name, root=None):
    if root:
        sys.path.append(root)
    tmp = importlib.import_module(name)
    if root:
        sys.path.remove(root)
    return tmp
