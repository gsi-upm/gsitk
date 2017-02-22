"""
Access operations on the feature extractors.
"""


import os
import glob
from gsitk.features import utils


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
        print(''.join(features))
    else:
        return features

