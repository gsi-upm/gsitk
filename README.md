# GSITK project

_gsitk_ is a library on top of scikit-learn that eases the development process on NLP machine learning driven projects.
It uses _numpy_, _pandas_ and related libraries to easy the development.

_gsitk_ manages datasets, features, classifiers and evaluation techniques, so that writing an evaluation pipeline results fast and simple.

# Installation and use

## Installation
_gsitk_ can be installed via pip, which is the recommended way:
```
pip install gsitk
```

Alternatively, gsitk can be installed by cloning this repository.

## Using _gsitk_

_gsitk_ saves into disk the datasets and some other necessary resources.
By default, all these data are stored in `/data`.
The environment variable `$DATA_PATH` can be set in order to specify an alternative directory.

# Feature extraction examples

## SIMON feature extractor
_gsitk_ includes the implementation of the SIMON feature extractor, presented in [this paper](https://doi.org/10.1016/j.knosys.2018.12.005).
To use it, two things are needed:

* A sentiment lexicon
* A word embeddings model that is [_gensim_](https://radimrehurek.com/gensim/) compatible.

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

# model also implements fit() and transform()
```


## Word2VecFeatures

This feature extractor implements the generic word vector model presented in [this paper](https://doi.org/10.1016/j.eswa.2017.02.002).
An example of use is shown below:

```python
from gsitk.features.word2vec import Word2VecFeatures


text = [
    ['my', 'cat', 'is', 'totally', 'happy'],
    ['my', 'dog', 'is', 'very', 'sad'],
]

# path is set to a Word2Vec model
# convolution parameter encondes pooling operation [average, maximum, minimum]

w2v_extractor = Word2VecFeatures(w2v_model_path=path, w2v_format='google_txt', convolution=[1,0,0])
X = model.transform(text)
# X is and array containing extrated features
```

# Cite 

In you use this module, please cite the following papers:

* Enhancing Deep Learning Sentiment Analysis with Ensemble Techniques in Social Applications

```
@article{ARAQUE2017236,
title = "Enhancing deep learning sentiment analysis with ensemble techniques in social applications",
journal = "Expert Systems with Applications",
volume = "77",
pages = "236 - 246",
year = "2017",
issn = "0957-4174",
doi = "https://doi.org/10.1016/j.eswa.2017.02.002",
url = "http://www.sciencedirect.com/science/article/pii/S0957417417300751",
author = "Oscar Araque and Ignacio Corcuera-Platas and J. Fernando Sánchez-Rada and Carlos A. Iglesias",
keywords = "Ensemble, Deep learning, Sentiment analysis, Machine learning, Natural language processing"
}
```

* A Semantic Similarity-based Perspective of Affect Lexicons for Sentiment Analysis

```
@article{ARAQUE2019346,
title = "A semantic similarity-based perspective of affect lexicons for sentiment analysis",
journal = "Knowledge-Based Systems",
volume = "165",
pages = "346 - 359",
year = "2019",
issn = "0950-7051",
doi = "https://doi.org/10.1016/j.knosys.2018.12.005",
url = "http://www.sciencedirect.com/science/article/pii/S0950705118305926",
author = "Oscar Araque and Ganggao Zhu and Carlos A. Iglesias",
keywords = "Sentiment analysis, Sentiment lexicon, Semantic similarity, Word embeddings",
}
```

# Support 

If you find bugs or want to make feature requests, please post an issue [here](https://github.com/gsi-upm/gsitk/issues/).
This project is under active development.

# Acknowledgements

This research work is supported by the EC through the H2020 project MixedEmotions (Grant Agreement no: 141111),
the Spanish Ministry of Economy under the R&D project Semola (TEC2015-68284-R)
and the project EmoSpaces (RTC-2016-5053-7); by ITEA3 project SOMEDI (15011);
and by MOSI-AGIL-CM (grant P2013/ICE-3019, co-funded by EU Structural Funds FSE and FEDER).
