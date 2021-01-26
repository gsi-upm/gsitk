# Classifiers

_gsitk_ offers a common interface, compatible with [scikit-learn predictors](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin) for implementing classifiers.
A classifier is a model that can be trained, or be already prepared to make predictions.

Currently, _gsitk_ has two classifier types:

* [`LexiconSum`](#use-of-lexicon): made for a simple use of an annotated lexicon.
* [`VaderClassifier`](#vader): wrapper for the popular [Vader](https://github.com/cjhutto/vaderSentiment) sentiment analysis classifer.


## Use of Lexicon

Performs a sum over the words of a given document, using the annotations from a lexicon. 
It follows the lexicon's annotation schema.
Normalizes the output to the range [-1, 0, 1].
The following example shows its use:

```python
from gsitk.classifiers import LexiconSum

# use a custom-lexicon
ls = LexiconSum({'good': 1, 'bad': -1, 'happy': 1, 'sad': -1, 'mildly': -0.1})

text = [
    ['my', 'dog', 'is', 'a', 'good', 'and', 'happy', 'pet'],
    ['my', 'cat', 'is', 'not', 'sad', 'just', 'mildly', 'bad'],
    ['not', 'happy', 'nor', 'sad'],
]

ls.predict(text)
```

```python
# output
array([ 1., -1.,  0.])
```

## Vader

Wrapper around the [implementation](https://github.com/cjhutto/vaderSentiment) of the original author.
This module does not need the text tokenized, as seen in the following example:


```python
from gsitk.classifiers import VaderClassifier

text = [
    'my dog is a good and happy pet',
    'my cat is not sad just mildly bad',
    'not happy nor sad',
]

VaderClassifier().predict(text)
```

```python
# output
array([ 1.,  0., -1.])
```
