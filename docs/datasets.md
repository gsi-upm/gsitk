# Datasets

_gsitk_ has a functionality suite for downloading, processing, and working with NLP datasets.
This allows researchers to work seamlessly with common datasets without delving into the details of data munging.

## Dataset Manager interface

Datasets can be accessed through the `DatasetManager`, an interface for dataset functionalities.
The manager is accessed in the following manner:

```python3
from gsitk.datasets.datasets import DatasetManager

dm = DatasetManager()
```

Dataset preparation includes downloading the data (if necessary) and pre-processing it.
This is the main functionality of the `DatasetManager`, and can be accessed in this way:

```python3
data = dm.prepare_datasets()
```

The `prepare_datasets` methods downloads **all** available datasets (if necessary) and pre-process them, loading them into memory.
Alternatively, it is possible to load a selection of datasets specifying theirs names:

```python3
data = dm.prepare_datasets(['vader', 'pl05'])
```

This example loads the [_vader_](https://github.com/cjhutto/vaderSentiment) and [_PL05_](http://www.cs.cornell.edu/people/pabo/movie-review-data/) datasets.
The `prepare_datasets` method returns a _dict_ that contains the datasets.
Each key corresponds the a dataset name, and the value is a [pandas Dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).


```python3
>>> data = dm.prepare_datasets(['vader', 'pl05'])

>>> type(data)
<class 'dict'>

>>> data.keys()
dict_keys(['vader', 'pl05'])

>>> type(data['vader'])
pandas.core.frame.DataFrame

>>> data['vader'].head()
   polarity                                               text
0         1  [somehow, i, was, blessed, with, some, really,...
1         1       [yay, ., another, good, phone, interview, .]
2         1  [we, were, number, deep, last, night, amp, the...
3         1            [lmao, allcaps, ,, amazing, allcaps, !]
4        -1  [two, words, that, should, die, this, year, :,...
```

Datasets are stored in pandas format, all operations are so you can make all pandas-related operations:

```python3
>>> data['vader']['polarity'].value_counts()
 1    2901
-1    1299
Name: polarity, dtype: int64
```

## Available datasets

Here we publish a list of the available datasets in _gsitk_.

* IMDB [`imdb`]
    * [Link](http://ai.stanford.edu/~amaas/data/sentiment/)
    * 50,000 sentiment analysis movie review instances, annotated with _negative_ and _positive_.
* IMDB un-supervised [`imdb_unsup`]
    * [Link](http://ai.stanford.edu/~amaas/data/sentiment/)
    * Additional unlabeled data, accompanying the IMDB dataset.
* Multi-Domain Sentiment Dataset (version 2.0) [`multidomain`]
    * [Link](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/)
    * Product review from amazon. There are several domains.
* PL04 [`pl04`]
    * [Link](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
    * 1000 positive and 1000 negative processed reviews. Introduced in Pang/Lee ACL 2004. 
* PL05 [`pl05`]
    * [Link](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
    * 5331 positive and 5331 negative processed sentences / snippets. Introduced in Pang/Lee ACL 2005.
* SemEval 2007 [`semeval07`]
    * Included in _gsitk_. No download necessary.
    * Affective Text task dataset. [Link](https://web.eecs.umich.edu/~mihalcea/affectivetext/).
    * Annotated with emotions (e.g. joy, fear, surprise) and polarity orientation (positive/negative).
* SemEval 2013 [`semeval13`]
    * For legal reasons, this dataset needs to be obtained by the author. _gsitk_ can process it then.
    * Sentiment analysis task datasets from SemEval 2013. [Link](https://www.cs.york.ac.uk/semeval-2013/task2.html)
* SemEval 2014 [`semeval14`]
    * For legal reasons, this dataset needs to be obtained by the author. _gsitk_ can process it then.
    * Sentiment analysis task datasets from SemEval 2014. [Link](http://alt.qcri.org/semeval2014/task9/)
* Sentiment140 [`sentiment140`]
    * [Link](http://help.sentiment140.com/)
    * Dataset with 1,6 million tweets annotated with sentiment.
* Stanford Sentiment Treebank (SST) [`sst`]
    * [Link](https://nlp.stanford.edu/sentiment/)
    * Detailed dataset with varied sentiment annotations.
* STS-Gold copurs [`sts`]
    * [Link](https://github.com/pollockj/world_mood/tree/master/sts_gold_v03)
    * Contains a dataset of tweets that have been human-annotated with sentiment labels. 
* Vader [`vader`]
    * [Link](https://github.com/cjhutto/vaderSentiment)
    * A dataset of tweets annotated with sentiment. Used for the creation of the _vader_ tools.

