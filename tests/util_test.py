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
import pytest

from gsitk.utils import Joiner

def test_join():
    joiner = Joiner()
    joined = joiner.fit_transform([
        ['the', 'cat', 'is', 'on', 'the', 'mat', '.'],
        ['my','dog','is','running','through','the','garden',',','he','is','so','happy','!','smile']
    ])
    result = [
        'the cat is on the mat .',
        'my dog is running through the garden , he is so happy ! smile'
    ]
    assert joined == result