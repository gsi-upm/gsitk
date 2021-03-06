#
# Copyright 2021 Grupo de Sistemas Inteligentes, DIT, Universidad Politecnica de Madrid (UPM)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Abstraction for embeddings methods, which wraps the common functionalities
used when performing embeddings operations.
"""

import numpy as np

from gsitk.features.features import Features


class Embedding(Features):
    """
    Abstraction for the embeddings operations.
    """
    def __init__(self, conv=[1, 0, 0]):
        self.conv = conv

    def build_vec(self, text):
        vecs = np.zeros((len(text), self.size))
        for i, word in enumerate(text):
            try:
                vecs[i] = self.model[word]
            except KeyError:
                continue
        if self.conv[0]:
            return np.average(vecs, axis=0)
        if self.conv[1]:
            return np.amax(vecs, axis=0)
        if self.conv[2]:
            return np.amin(vecs, axis=0)

    def comments2vec(self, text):
        """
        Convert to vecs the text passed as a iterator.
        The conv parameter defines the convolutional function to use:
        [x,x,x] <=> [average, max, min]
        Only one convolutional function by pass is supported.
        We consider each one a different type of feature.
        """
        l = len(text)
        vecs = np.zeros((l, self.size))
        for i in range(l):
            vecs[i] = self.build_vec(text=text[i])
        return vecs

    def check_vector(self, v):
        if np.any(np.isnan(v)):
            v_ = np.zeros((v.shape[0],))
        else:
            v_ = v
        return v_


            
