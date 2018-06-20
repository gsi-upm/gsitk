# GSITK project

_gsitk_ is a library on top of scikit-learn that eases the development process on NLP machine learning driven projects.
It uses _numpy_, _pandas_ and related libraries to easy the development.

_gsitk_ manages datasets, features, classifiers and evaluation techniques, so that writing an evaluation pipeline results fast and simple.

# Feature extractors

## SIMON feature extractor
_gsitk_ includes the implementation of the SIMON feature extractor.
To use it, two things are needed:
- A sentiment lexicon
- A word embeddings model that is _gensim_ compatible.

For example, using only the lexicon from [Bing Liu](https://dl.acm.org/citation.cfm?id=1014073) and a [embeddings model](https://code.google.com/archive/p/word2vec/) that is in the current directory:

```python
from gsitk.features import simon
from nltk.corpus import opinion_lexicon
from gensim.models.keyedvectors import KeyedVectors

lexicon = [list(opinion_lexicon.positive()), list(opinion_lexicon.negative())]

embedding_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

simon_transformer = simon.Simon(lexicon=lexicon, n_lexicon_words=200, embedding=embedding_model)

# simon_transformer has the fit() and transform() methods, so it can be used in a Pipeline
```

To enhance performance, it is recommendable to use a more complete scikit-learn pipe that implements normalization and feature selection in conjuction with the SIMON feature extraction.

```python
from gsitk.features import simon

simon_model = simon.Simon(lexicon=lexicon, n_lexicon_words=200, embedding=embedding_model)
model = simon.simon_pipeline(simon_transformer=simon_model, percentile=25)

# model also implemtens fit() and transform()
```