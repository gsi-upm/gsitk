"""
Utilities that are commonly used.
"""

from sklearn.base import TransformerMixin

class JoinTransformer(TransformerMixin):
    
    def transform(self, X, y=None, **fit_params):
        return [' '.join(x)for x in X]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
