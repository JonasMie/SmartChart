import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Imputer, StandardScaler
from sklearn.utils import check_array
from sknn.mlp import Classifier, Layer

import learning.utils as utils

booleans = ("track_genre_electronic", "track_genre_pop", "track_genre_hiphop", "track_genre_rock", "track_genre_other",
            "artist_genre_electronic", "artist_genre_pop", "artist_genre_hiphop", "artist_genre_rock",
            "artist_genre_other" "available_on_spotify_in_ger", "exists_remix", "is_2010s", "is_2000s", "is_1990s",
            "is_1980s", "is_other_decade", "is_male", "is_female", "is_group", "german", "american", "other_country",
            "mean_chart_peak_0", "mean_chart_peak_1", "mean_chart_peak_2", "mean_chart_peak_3", "mean_chart_peak_4",
            "mean_chart_peak_5", "mean_chart_peak_6", "mean_album_chart_peak_0", "mean_album_chart_peak_1",
            "mean_album_chart_peak_2", "mean_album_chart_peak_3", "mean_album_chart_peak_4", "mean_album_chart_peak_5",
            "mean_album_chart_peak_6")


class CustomFunctionTransformer(FunctionTransformer):
    def __init__(self, func):
        super(CustomFunctionTransformer, self).__init__(func=func)
        self.predict_proba = None

    def fit(self, X, y=None):
        if self.validate:
            check_array(X, self.accept_sparse, force_all_finite=False)
        return self

    def transform(self, X, y=None):
        if self.validate:
            check_array(X, self.accept_sparse, force_all_finite=False)
        func = self.func

        return func(X, *((y,) if self.pass_y else ()))


n = 0


def get_booleans(X):
    bools = [x for x in X.columns.values if x in booleans]
    return np.array(X[bools])


def get_continuous(X):
    continuous = [x for x in X.columns.values if x not in booleans]
    return np.array(X[continuous])


def getPipeline(data, classifier):
    features = FeatureUnion(
            [
                ('booleanPipeline', Pipeline([
                    ('boolean_transformer', CustomFunctionTransformer(get_booleans)),
                    ('bool_imputer', Imputer(strategy="most_frequent")),
                ])),
                ('continousPipeline', Pipeline([
                    ('continousSelector', CustomFunctionTransformer(get_continuous)),
                    ('cont_imputer', Imputer()),
                    ('scaler', StandardScaler()),
                ])),
            ])
    return Pipeline([
        ('features', features),
        ('neural network', classifier)
    ])


def getClassifier(units=15, learning_rate=.001, n_iter=25):
    return Classifier(
            layers=[
                Layer(type="Sigmoid", units=units),
                Layer(type="Softmax")],
            learning_rate=learning_rate,
            n_iter=n_iter)


def getData(size, ratio, features):
    complete_data = utils.getData(size, split=False)

    if features is not None:
        features.append('peak_cat')
        complete_data = complete_data[features]
    threshold = int(size * ratio)

    training_data = complete_data[:threshold]
    test_data = complete_data[threshold:]

    training_targets = training_data['peak_cat']
    training_data.drop('peak_cat', axis=1, inplace=True)

    test_targets = test_data['peak_cat']
    test_data.drop('peak_cat', axis=1, inplace=True)

    return training_data, training_targets, test_data, test_targets


def train(size, ratio, units, learning_rate, iterations, features):
    training_data, training_targets, test_data, test_targets = getData(size, ratio, features)
    classifier = getClassifier(units, learning_rate, iterations)
    pipeline = getPipeline(training_data, classifier)

    clf = pipeline.fit(training_data, training_targets)

    print cross_val_score(clf, training_data, training_targets)