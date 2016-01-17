from sklearn.pipeline import Pipeline, FeatureUnion
from sknn.mlp import Classifier, Layer

import learning.utils as utils

booleans = ["track_genre_electronic", "track_genre_pop", "track_genre_hiphop", "track_genre_rock", "track_genre_other",
            "artist_genre_electronic", "artist_genre_pop", "artist_genre_hiphop", "artist_genre_rock",
            "artist_genre_other" "available_on_spotify_in_ger", "exists_remix", "is_2010s", "is_2000s", "is_1990s",
            "is_1980", "is_other_decade", "is_male", "is_female", "is_group", "german", "american", "other_country",
            "mean_chart_peak_0", "mean_chart_peak_1", "mean_chart_peak_2", "mean_chart_peak_3", "mean_chart_peak_4",
            "mean_chart_peak_5", "mean_chart_peak_6", "mean_album_chart_peak_0", "mean_album_chart_peak_1",
            "mean_album_chart_peak_2", "mean_album_chart_peak_3", "mean_album_chart_peak_4", "mean_album_chart_peak_5",
            "mean_album_chart_peak_6"],


class ColumnExtractor(object):
    def __init__(self, is_bool):
        self.is_bool = is_bool

    def transform(self, X):
        print "jo"
        return []
        if self.is_bool:
            columns = [x for x in X.columns.values if x in booleans]
        else:
            columns = [x for x in X.columns.values if x not in booleans]
        return X[columns]

    def fit(self, X, y=None):
        print self, X
        return self


def getPipeline(data, classifier):
    boolean = []
    continuous = []

    features = FeatureUnion(
            [
                ('booleanPipeline', Pipeline([
                    ('boolean_transformer', ColumnExtractor(True)),
                    # ('bool_imputer', Imputer(strategy="most_frequent")),
                    # ('oneHotEncoder', OneHotEncoder())
                ])),
                # ('continousPipeline', Pipeline([
                #     ('continousSelector', ColumnExtractor(cols=continuous)),
                #     ('cont_imputer', Imputer()),
                #     ('scaler', StandardScaler())
                # ])),

                # ('encodedPipeline', Pipeline([
                #     ('encodedSelector', ColumnExtractor(cols=encoded)),
                #
                # ])),
            ])
    return Pipeline([
        # ('features', features),
        ('boolean_transformer', ColumnExtractor(True)),
        # ('neural network', classifier)
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
