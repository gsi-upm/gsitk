from sklearn.base import TransformerMixin
import numpy as np

class Preprocesser(TransformerMixin):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_trans = []
        for i, x_i in enumerate(X):
            X_trans.append(self.preprocessor.preprocess(x_i))
        return np.array(X_trans)

class JoinTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return [' '.join(x) for x in X]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
