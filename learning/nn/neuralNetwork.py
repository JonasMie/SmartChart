import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
from sknn.mlp import Classifier, Layer


class ColumnExtractor(object):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c + 1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self


def getPipeline(boolean, continous, encoded, classifier, n_iter=25):
    features = FeatureUnion(
            [
                ('booleanPipeline', Pipeline([
                    ('booleanSelector', ColumnExtractor(cols=boolean)),
                    ('oneHotEncoder', OneHotEncoder())
                ])),
                ('continousPipeline', Pipeline([
                    ('continousSelector', ColumnExtractor(cols=continous)),
                    ('scaler', StandardScaler())
                ])),
                ('encodedPipeline', Pipeline([
                    ('encodedSelector', ColumnExtractor(cols=encoded)),

                ])),
            ])
    return Pipeline([('imputer', Imputer()), ('features', features),
                     ('neural network', classifier)])


def getClassifier(units=100, pieces=2, learning_rate=.001, n_iter=25):
    return Classifier(
            layers=[
                Layer("Maxout", units=100, pieces=2),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=25)

