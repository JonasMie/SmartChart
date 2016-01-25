from __future__ import division

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Imputer, StandardScaler
from sklearn.utils import check_array
from sknn.mlp import Classifier, Layer

import learning.utils as utils

# ignore annoying pandas warning
pd.options.mode.chained_assignment = None

train_errors = None
valid_errors = None
prev_train_error = 0
prev_valid_error = 0
plot = None
best_train_i = None
best_valid_i = None

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


def getNet(units, learning_rate, n_iter, learning_rule, batch_size, weight_decay, dropout_rate, loss_type, n_stable,
           debug, verbose, callbacks, valid_size):
    layers = [Layer(type="Sigmoid", units=u) for u in units]
    layers.append(Layer(type="Softmax"))
    return Classifier(
            layers=layers,
            learning_rate=learning_rate,
            n_iter=n_iter,
            learning_rule=learning_rule,
            batch_size=batch_size,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            loss_type=loss_type,
            n_stable=n_stable,
            debug=debug,
            verbose=verbose,
            callback=callbacks,
            # valid_set=valid_set
            valid_size=valid_size
    )


def getData(size, ratio, features, balanced):
    complete_data = utils.getData(size, split=False, balanced=balanced)

    if features is not None and len(features) > 0:
        features.append('peak_cat')
        complete_data = complete_data[features]
    threshold = int(complete_data.shape[0] * (1 - ratio))

    training_data = complete_data[:threshold]
    test_data = complete_data[threshold:]

    training_targets = training_data['peak_cat']
    training_data.drop('peak_cat', axis=1, inplace=True)

    test_targets = test_data['peak_cat']
    test_data.drop('peak_cat', axis=1, inplace=True)

    return training_data, training_targets, test_data, test_targets


def on_train_start(**variables):
    pass
    # main_utils.startProgress("Beginning training")


def on_train_finish(**variables):
    global plot
    if plot is not None:
        utils.plot_lines(data=[train_errors, valid_errors], labels=["Training error", "Validation error"],
                         xlabel="number of epochs",
                         ylabel=config['loss_type'],
                         title="trainng and validation error", suptitle=None, conf=config, additionals=[
                [best_train_i, variables['best_train_error']], [best_valid_i, variables['best_valid_error']]],
                         path=plot)


def on_epoch_start(**variables):
    pass


def on_epoch_finish(**variables):
    global config, train_errors, valid_errors, best_train_i, best_valid_i
    train_errors[variables['i'] - 1] = variables['avg_train_error']
    valid_errors[variables['i'] - 1] = variables['avg_valid_error']
    if variables['is_best_train']:
        best_train_i = variables['i']
    if variables['is_best_valid']:
        best_valid_i = variables['i']
        # main_utils.progress(variables['i'] / float(config['epochs']) * 100)


def on_batch_start(**variables):
    pass


def on_batch_finish(**variables):
    pass


default_callbacks = {
    'on_train_start': on_train_start, 'on_epoch_start': on_epoch_start,
    'on_batch_start': on_batch_start, 'on_epoch_finish': on_epoch_finish,
    'on_train_finish': on_train_finish, 'on_batch_finish': on_batch_finish
}


def train(conf, plot_path, debug, verbose, callbacks=default_callbacks):
    global config, plot, train_errors, valid_errors
    plot = plot_path
    train_errors = np.zeros(conf['epochs'])
    valid_errors = np.zeros(conf['epochs'])
    training_data, training_targets, valid_data, valid_targets = getData(conf['datasets'], 0, conf['features'],
                                                                         balanced=conf['balanced'])

    # if there is not enough data available
    conf['datasets'] = training_data.shape[0]
    if conf['units'] is None:
        conf['units'] = [training_data.shape[1]]
    config = conf
    net = getNet(conf['units'], conf['learning_rate'], conf['epochs'], conf['learning_rule'],
                 conf['batch_size'], conf['weight_decay'], conf['dropout_rate'],
                 conf['loss_type'], n_stable=conf['n_stable'], debug=debug, verbose=verbose, callbacks=callbacks,
                 # valid_set=(valid_data, valid_targets)
                 valid_size=conf['ratio']
                 )

    pipeline = getPipeline(training_data, net)
    return pipeline.fit(training_data, training_targets)


def predict(track_id, clf):
    data = utils.selectData(track_id)
