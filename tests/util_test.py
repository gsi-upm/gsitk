import pytest

from gsitk.utils import JoinTransformer

def test_join():
    joiner = JoinTransformer()
    joined = joiner.fit_transform([
        ['the', 'cat', 'is', 'on', 'the', 'mat', '.'],
        ['my','dog','is','running','through','the','garden',',','he','is','so','happy','!','smile']
    ])
    result = [
        'the cat is on the mat .',
        'my dog is running through the garden , he is so happy ! smile'
    ]
    assert joined == result