import pytest

from gsitk.preprocess import normalize

@pytest.fixture
def text_df():
    import pandas as pd
    with open('tests/data/test_dataset.txt') as f:
        text = f.readlines()
    df = pd.DataFrame(columns=['text'],
                      data=text)
    return df


def test_normalize_text(text_df):
    norm = normalize.normalize_text(text_df)
    assert norm[0] ==  ['the', 'cat', 'is', 'on', 'the', 'mat', '.']
    assert norm[1] == ['my','dog','is','running','through','the','garden',',','he','is','so','happy','!','smile']
