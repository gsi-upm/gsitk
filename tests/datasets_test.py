from gsitk.datasets import (
    sentiment140
)

def test_sentiment140():
    data = sentiment140.prepare_data(download=False)
    assert data['polarity'].value_counts().index[0] == -1
    assert data['polarity'].value_counts().values[0] == 10
    assert (data['text'].apply(len) > 0).all()
