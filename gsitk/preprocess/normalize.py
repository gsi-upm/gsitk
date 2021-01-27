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
Normalize text
"""

import string
from nltk import word_tokenize

from gsitk.preprocess.pprocess_twitter import tokenize

noise = set(string.punctuation) - set('¡!¿?,.:')  # > and < are removed also
noise = {ord(c): None for c in noise}

def _normalize_text(text):
        t = tokenize(text)
        t = t.lower().translate(noise)
        return word_tokenize(t)

def normalize_text(data):
    # Tokenize and clean the test
    text_data = data['text'].apply(_normalize_text)
    return text_data

def preprocess(text):
    return _normalize_text(text)
