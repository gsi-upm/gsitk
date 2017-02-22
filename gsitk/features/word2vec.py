"""
Feature extraction with Word2Vec, as explained in

"Enhancing deep learning sentiment analysis with ensemble techniques in social applications",
http://dx.doi.org/10.1016/j.eswa.2017.02.002


Needs a Word2Vec model previously trained. Compatible with Gensim and Google word2vec format.
"""

import os
import numpy as np
from gensim.models import Word2Vec


def build_vec(text, model, conv):
    vecs = list()
    for word in text:
        try:
            v = model[word]
            vecs.append(v)
        except KeyError:
            continue
    if conv[0]:
        return np.average(vecs, axis=0)
    if conv[1]:
        return np.amax(vecs, axis=0)
    if conv[2]:
        return np.amin(vecs, axis=0)


def comments2vec(x, size, model, conv=[1, 0, 0]):
    """Convert to vecs the text passed as a iterator.
    The conv parameter defines the convolutional function to use:
    [x,x,x] <=> [average, max, min]
    Only one convolutional function by pass is supported.
    We consider each one a different type of feature.
    """
    l = len(x)
    vecs = np.zeros((l, size))
    for i in range(l):
        vecs[i] = build_vec(text=x[i], model=model, conv=conv)
    return vecs


def check_vector(v):
    if np.any(np.isnan(v)):
        v_ = np.zeros((v.shape[0],))
    else:
        v_ = v
    return v_


def load_w2v(w2v_model_path, w2v_format):
    if not os.path.exists(w2v_model_path):
        raise ValueError("Word2Vec model path does not exist.")

    if w2v_format == 'gensim':
        w2v = Word2Vec.load(w2v_model_path)
    elif w2v_format == 'google_txt':
        w2v = Word2Vec.load_word2vec_format(w2v_model_path, binary=False)
    elif w2v_format == 'google_bin':
        w2v = Word2Vec.load_word2vec_format(w2v_model_path, binary=True)
    else:
        raise ValueError("w2v_format={} is not valid.".format(w2v_format))

    return w2v


def transform(X, w2v_model_path, w2v_format='gensim', convolution=[1,0,0]):
    """Extract the features.
    This considers X to be a list of lists of texts.
    [
        ['my', 'dog', 'run', 'in', 'the', 'rain']
    ]

    w2v_format can be 'gensim', 'google_txt' or 'google_bin'
    """
    w2v = load_w2v(w2v_model_path, w2v_format)

    vecs = comments2vec(x=X, size=w2v.vector_size, model=w2v, conv=convolution)

    vecs = check_vector(vecs)

    return vecs
