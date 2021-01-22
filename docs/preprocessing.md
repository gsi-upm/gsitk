# Pre-processing

As a NLP-centered software, _gsitk_ has various functionalities for pre-processing tasks.
Pre-processing is a central part of data munging in NLP, and more specifically, in sentiment analysis.
In order to ease this kind of operation, _gsitk_ offers the following type of pre-processers:

* **Simple**: the simple and more efficient pre-processor. Indicated for processing large datasets, as it is the fastest. Based on regular expressions that parse English text.
* **Pre-process Twitter**: a processor indicated for parsing Twitter text. Also based on regular expressions, extracts common emoji as special tokens, and transforms hashtags and mentions, normalizing them.
* **Normalize**: An all-purpose pre-processor. It is not as efficient as the other options, and performs word tokenization based on NLTK.

## Direct use

The more straight-forward way to use the pre-processing utilities is presented as follows:

```python3
from gsitk.preprocess import simple, pprocess_twitter, normalize

text = "My grandmother is an apple. Please, believe me!"
twitter_text = "@POTUS please let me enter to the USA #thanks"

print('simple', simple.preprocess(text))
print('twitter', pprocess_twitter.preprocess(twitter_text))
print('normalize', normalize.preprocess(text))
```

These lines of code would output the following. Note how the Twitter data is transformed, normalizing the mention to an user and a hashtag.

```
simple ['my', 'grandmother', 'is', 'an', 'apple', '.', 'please', ',', 'believe', 'me', '!']
twitter <user> please let me enter to the usa <allcaps> <hastag> thanks
normalize ['my', 'grandmother', 'is', 'an', 'apple', '.', 'please', ',', 'believe', 'me', '!']
```

## Preprocesser interface

To facilitate the use of the preprocessing functions, _gsitk_ offers an interface that is compatible with [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html): the `Preprocesser`.
A simple script using this interface is:

```python3
from gsitk.preprocess import pprocess_twitter, Preprocesser

texts = [
    "@POTUS please let me enter to the USA #thanks",
    "If only Bradley's arm was longer. Best photo ever. #oscars"
]
Preprocesser(pprocess_twitter).transform(texts)
```

```
# output:
array(['<user> please let me enter to the usa <allcaps> <hastag> thanks',
       "if only bradley's arm was longer. best photo ever. <hastag> oscars"],
      dtype='<U66')
```

The compatibility with scikit Pipelines can be used to ease the use of preprocessing.
For example:

```python3
from sklearn.pipeline import Pipeline
from gsitk.preprocess import normalize, Preprocesser, JoinTransformer

texts = [
    "This cat is crazy, he is not on the mat!",
    "Will no one rid me of this turbulent priest?"
]

preprocessing_pipe = Pipeline([
    ('twitter', Preprocesser(normalize)),
    ('join', JoinTransformer())
])

preprocessing_pipe.fit_transform(texts)
```

```
# output:
['this cat is crazy , he is not on the mat !',
 'will no one rid me of this turbulent priest ?']
```

## Stop word removal

Removing stop words is a pervasive task in NLP.
_gsitk_ includes a functionality for this, using the stop word collections in NLTK.

```python3
from gsitk.preprocess.stopwords import StopWordsRemover

texts = [
    "this cat is crazy , he is not on the mat !",
    "will no one rid me of this turbulent priest ?"
]

StopWordsRemover().fit_transform(texts)
```

```
# output:
['cat crazy , mat !', 'one rid turbulent priest ?']
```

As it uses the NLTK stop word collections, several languages can be parsed, as in this Spanish example.

```python3
from gsitk.preprocess.stopwords import StopWordsRemover

texts = [
    "entre el clavel blanco y la rosa roja , su majestad escoja",
    "con diez cañones por banda viento en popa a toda vela",
]

StopWordsRemover(language='spanish').fit_transform(texts)
```

```
# output:
['clavel blanco rosa roja , majestad escoja',
 'diez cañones banda viento popa toda vela']
```

## Embeddings trick

The paper _DepecheMood++: a Bilingual Emotion Lexicon Built Through Simple Yet Powerful Techniques_ ([link here](https://doi.org/10.1109/TAFFC.2019.2934444)) introduces a technique called the **Embeddings trick**.
In short, it consists on replacing certain words by others using a word embedding model.
It is used to expand an existing emotion dictionary.
In any way, we consider it is an useful technique, and has been implemented in _gsitk_, making it easier to replicate the mentioned paper.


TODO

