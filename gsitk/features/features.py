"""
Access operations on the feature extractors.
"""


import os
import glob
from gsitk.features import utils


class Features():
    """
    Class that abstracts the features. 
    """
    def transform(self, X):
        """
        Transform the text (should be normalized) to numeric features.
        Must be implemented by the class that inherits.
        """
        pass 


def load_features(name, format=None):
    """Reads the features."""
    return utils.read_features(name, format)


def save_features(features, name):
    utils.save_features(features, name)


def view_features(pprint=True):
    """
    Check the available features, the ones that have already been
    extracted and stored.
    """
    features = []
    for feats in glob.glob(utils.features_path + '*'):
        filename = os.path.basename(feats)
        name = os.path.splitext(filename)[0]
        format = utils.detect_saving_format(filename)
        features.append(utils._check_features(name, format))

    if pprint:
        print('\n'.join(features))
    else:
        return features


