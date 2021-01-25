# Evaluation


_gsitk_ implements several functionalities for managing complex evaluation scenarios in Sentiment Analysis.
Evaluating new models, and comparing them against previous works is the bread and butter of progress in Sentiment Analysis (see [this](http://nlpprogress.com/english/sentiment_analysis.html), por example).
This way of progress makes the technicalities of the evaluation difficult to replicate, and costly to perform.
In this sense, _gsitk_ offers a number of functionalities to aid practitioners and researches in performing evaluation in the field of Sentiment Analysis.

We divide the evaluation documentation in two:

* [Basic Evaluation](#basic-evaluation)
* [Advanced Evaluation](#advanced-evaluation)
 
## Basic Evaluation

As many of _gsitk_'s modules, the evaluation methods are compatible with scikit-learn's [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline).
In fact, the we work with the evaluation interfaces as we do with these Pipelines.
Let's perform a simple evaluation with _gsitk_ from the beginning.
First, we load the evaluation datasets:

```python
from gsitk.datasets.datasets import DatasetManager

dm = DatasetManager()
data = dm.prepare_datasets(['vader', 'pl05'])
```

Following, we define the models we want to evaluate.
We define two pipelines (`pipeline` and `pipeline2`) that use scikit components.
Observe that we are naming the pipelines and the pipelines' steps through the `name` property. 
This will be useful to locate each model.
For `pipeline2`, we do not name the pipeline's steps.


```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

pipeline.fit(data_ready['vader']['text'].values,
             data_ready['vader']['polarity'].values.astype(int))
pipeline.name = 'pipeline_trained'
pipeline.named_steps['vect'].name = 'myvect'
pipeline.named_steps['tfidf'].name = 'mytfidf'
pipeline.named_steps['clf'].name = 'mylogisticregressor'


pipeline2 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

pipeline2.fit(data_ready['pl05']['text'].values,
              data_ready['pl05']['polarity'].values.astype(int))
pipeline2.name = 'pipeline_trained2'
```

At this point, we may need to adapt the datasets so that the text is represented by a string, not a list of tokens:

```python
data_ready = {}
for data_k, data_v in data.items():
    data_ready[data_k] = data_v.copy()
    data_ready[data_k]['text'] = data_v['text'].apply(' '.join).values
```

At this point we are ready to perform the evaluation.
For this purpose, we use the `Evaluation` class, which is defined by the following parameters:

* `tuples`: used in [Advanced Evaluation](#advanced-evaluation).
* `datasets`: a `dict` of the datasets to use for the evaluation.
* `pipelines`: a `list` of scikit pipelines that represent the models to evaluate.


```python
from gsitk.evaluation.evaluation import Evaluation

# define datasets for evaluation
datasets_evaluation = {
    'vader': data_ready['vader'],
    'pl05': data_ready['pl05']
}

# configure evaluation
ev = Evaluation(tuples=None,
                datasets=datasets_evaluation,
                pipelines=[pipeline, pipeline2])
                
# perform evaluation, this can take a little long
ev.evaluate()

# results are stored in ev, and are in pandas DataFrame format
ev.results
```

The output is shown below:
```
|   | Dataset | Features | Model                    | CV    | accuracy | precision_macro | recall_macro | f1_weighted | f1_micro | f1_macro | Description                                       |
|---|---------|----------|--------------------------|-------|----------|-----------------|--------------|-------------|----------|----------|---------------------------------------------------|
| 0 | vader   | None     | pipeline_trained__vader  | False | 0.992143 | 0.992596        | 0.988998     | 0.992128    | 0.992143 | 0.990772 | vect(myvect) --> tfidf(mytfidf) --> clf(mylogi... |
| 1 | vader   | None     | pipeline_trained2__vader | False | 0.596429 | 0.630961        | 0.649194     | 0.608576    | 0.596429 | 0.59155  | vect --> tfidf --> clf                            |
| 2 | pl05    | None     | pipeline_trained__pl05   | False | 0.578962 | 0.585842        | 0.579002     | 0.570405    | 0.578962 | 0.570422 | vect(myvect) --> tfidf(mytfidf) --> clf(mylogi... |
| 3 | pl05    | None     | pipeline_trained2__pl05  | False | 0.926788 | 0.926838        | 0.926787     | 0.926786    | 0.926788 | 0.926786 | vect --> tfidf --> clf                            |
```

In the results table we can observe how the designed models obtain different metrics in the evaluation datasets.
Also, the names we used when defining the models are used to identify each pipeline.


When configured as in the example, `Evaluation` uses already trained models to predict on the defined datasets.
For a more configurable framework, see [Advanced Evaluation](#advanced-evaluation).

## Advanced Evaluation 

Of course, more complex evaluation methods normally need to be done.
For this, _gsitk_'s evaluation framework has a more advanced interface.

Internally, the evaluation process uses evaluation tuples (`EvalTuple`), which are a method for specifying which datasets, features and classifiers we want to evaluate. For evaluating a set of models that predict from a set of features, a `EvalTuple`` are specified. The next example evaluates a simple logistic regressions model that uses word2vec features to predict the sentiment of the IMDB dataset.

We prepare the data and extract the features we want to evaluate:

```python

from gsitk.datasets.datasets import DatasetManager
from gsitk.features.word2vec import Word2VecFeatures

dm = DatasetManager()
data = dm.prepare_datasets(['imdb',])

w2v_feat = Word2VecFeatures(w2v_model_path='/data/w2vmodel_500d_5mc')
transformed = w2v_feat.transform(data['imdb']['text'].values)
```

We define the machine learning model to use:

```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()

# transformed is the features extracted from the IMDB dataset
# to properly evaluate, separate in train and test 
# using the original dataset fold
train_indices = (data['imdb']['fold'] == 'train').values
test_indices =(data['imdb']['fold'] == 'test').values

transformed_train = transformed[train_indices]
transformed_test = transformed[test_indices]


sgd.fit(transformed_train, data['imdb']['polarity'][train_indices])
```

After this, we prepare the model, features and EvalTuple for the evaluation:

```python
from gsitk.pipe import Model, Features, EvalTuple

models = [Model(name='sgd', classifier=sgd)]

feats = [Features(name='w2v__imdb_test', dataset='imdb', values=transformed_test)]

ets = [EvalTuple(classifier='sgd', features='w2v__imdb_test', labels='imdb')]
```

Finally, we just need to perform the evaluation:

```python
from gsitk.evaluation.evaluation import Evaluation

ev = Evaluation(datasets=data, features=feats, models=models, tuples=ets)

# run the evaluation
ev.evaluate()

# view the results
ev.results
```

The output is:

```
|   | Dataset | Features       | Model | CV    | accuracy | precision_macro | recall_macro | f1_weighted | f1_micro | f1_macro |
|---|---------|----------------|-------|-------|----------|-----------------|--------------|-------------|----------|----------|
| 0 | imdb    | w2v__imdb_test | sgd   | False | 0.76164  | 0.782904        | 0.76164      | 0.757075    | 0.76164  | 0.757075 |
```

From the output, we can see how the evaluation has been done.
Through the shown tools, we can define more complex evaluation procedures, adapting to the needs of practitioners and researchers.

If we want to perform [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), there is a specific evaluation tuple for that: the `CrossEvalTuple`, which has the same interface as the `EvalTuple`.

---

The full example is shown below:

```python
from gsitk.datasets.datasets import DatasetManager
from gsitk.features.word2vec import Word2VecFeatures
from sklearn.linear_model import SGDClassifier
from gsitk.pipe import Model, Features, EvalTuple
from gsitk.evaluation.evaluation import Evaluation

dm = DatasetManager()
data = dm.prepare_datasets(['imdb',])


w2v_feat = Word2VecFeatures(w2v_model_path='/data/w2vmodel_500d_5mc')
transformed = w2v_feat.transform(data['imdb']['text'].values)


sgd = SGDClassifier()

# transformed is the features extracted from the IMDB dataset
# to properly evaluate, separate in train and test 
# using the original dataset fold
train_indices = (data['imdb']['fold'] == 'train').values
test_indices =(data['imdb']['fold'] == 'test').values

transformed_train = transformed[train_indices]
transformed_test = transformed[test_indices]

sgd.fit(transformed_train, data['imdb']['polarity'][train_indices])

models = [Model(name='sgd', classifier=sgd)]
feats = [Features(name='w2v__imdb_test', dataset='imdb', values=transformed_test)]
ets = [EvalTuple(classifier='sgd', features='w2v__imdb_test', labels='imdb')]

ev = Evaluation(datasets=data, features=feats, models=models, tuples=ets)

# run the evaluation
ev.evaluate()

# view the results
ev.results
```

