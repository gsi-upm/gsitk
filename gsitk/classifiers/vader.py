import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import ClassifierMixin

polarity_transform = {0: 1, 1: 0, 2: -1}

class VaderClassifier(ClassifierMixin):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X, y=None):
        y_hat = np.zeros((X.shape[0]))
        for i, x_i in enumerate(X):
            pol = self.sid.polarity_scores(x_i)
            pred = np.array([pol['pos'], pol['neu'], pol['neg']]).argmax()
            y_hat[i] = polarity_transform[pred]

        return np.array(y_hat)