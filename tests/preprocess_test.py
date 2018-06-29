import pytest

from gsitk.preprocess import normalize, simple, Preprocesser, embeddings_trick, stopwords


@pytest.fixture
def text_df():
    import pandas as pd
    with open('tests/data/test_dataset.txt') as f:
        text = f.readlines()
    text = [t.strip() for t in text]
    df = pd.DataFrame(columns=['text'],
                      data=text)
    return df


def test_normalize_text(text_df):
    norm = normalize.normalize_text(text_df)
    assert norm[0] ==  ['the', 'cat', 'is', 'on', 'the', 'mat', '.']
    assert norm[1] == ['my','dog','is','running','through','the','garden',',','he','is','so','happy','!','smile']


def test_clean_str(text_df):
    clean = simple.normalize_text(text_df)
    assert clean[0] ==  ['the', 'cat', 'is', 'on', 'the', 'mat', '.']
    assert clean[1] == ['my','dog','is','running','through','the','garden',',','he','is','so','happy','!',')']


def test_preprocesser(text_df):
    # test preprocesser with normalize functionality
    pp = Preprocesser(normalize)
    preprocessed = pp.fit_transform(text_df['text'].values)
    assert preprocessed[0] ==  ['the', 'cat', 'is', 'on', 'the', 'mat', '.']
    assert preprocessed[1] == ['my','dog','is','running','through','the','garden',',','he','is','so','happy','!','smile']

    # test preprocesser with simple functionality
    pp = Preprocesser(simple)
    preprocessed = pp.fit_transform(text_df['text'].values)
    assert preprocessed[0] ==  ['the', 'cat', 'is', 'on', 'the', 'mat', '.']
    assert preprocessed[1] == ['my','dog','is','running','through','the','garden',',','he','is','so','happy','!',')']


def test_embeddings_tricker():
    et = embeddings_trick.EmbeddingsTricker(model_path='tests/data/w2v_model', w2v_format='google_txt',
                                            vocabulary=['to', 'the', 'repeat', 'allcaps'])
    closest = et.closest_word('user')
    assert isinstance(closest, str)
    assert len(closest) > 0

    trans = et.fit_transform([['.', 'user', 'i', '!', 'to', 'the', ',', 'repeat',],
                              [ 'the', ',', 'repeat', 'allcaps', 'a', 'my', 'and', 'you']])

    assert trans == [['!', ',', 'it', '.', 'to', 'the', 'and', 'repeat'],
                     ['the', 'and', 'repeat', 'allcaps', 'the', 'the', ',', 'it']]


def test_stopwords_remover(text_df):
    stop = stopwords.StopWordsRemover(type='nltk', language='english')
    trans = stop.fit_transform(text_df['text'].values)
    assert trans == ['The cat mat.', 'My dog running garden, happy! :)'] 