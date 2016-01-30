from __future__ import division

import math

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Imputer, StandardScaler, MinMaxScaler
from sklearn.utils import check_array
from sknn.mlp import Classifier, Layer
from termcolor import colored

import learning.utils as utils

# ignore annoying pandas warning
pd.options.mode.chained_assignment = None

config = None
train_errors = None
valid_errors = None
prev_train_error = 0
prev_valid_error = 0
plot = None
best_train_i = None
best_valid_i = None
unit_iter_train_error = []
unit_iter_valid_error = []

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
    global config
    union = [
        ('continousPipeline', Pipeline([
            ('continousSelector', CustomFunctionTransformer(get_continuous)),
            ('cont_imputer', Imputer()),
            ('scaler', StandardScaler()),
        ])),
    ]

    if [i for i in data if i in booleans]:
        strat = "most_frequent"
        # strat = "mean"
        union.append(
                ('booleanPipeline', Pipeline([
                    ('boolean_transformer', CustomFunctionTransformer(get_booleans)),
                    # ('boolean_scaler', MinMaxScaler(feature_range=(-1,1))),
                    ('bool_imputer', Imputer(strategy=strat)),
                ])),
        )

    features = FeatureUnion(union)
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


def getData(size, ratio, features, balanced, type):
    complete_data = utils.getData(size, split=False, balanced=balanced, type=type)

    if features is not None and len(features) > 0 and type != "md" and type != "mir":
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
                         title="training and validation error", suptitle=None, conf=config, additionals=[
                [best_train_i, variables['best_train_error']], [best_valid_i, variables['best_valid_error']]],
                         path=plot)
    if config['unit_range']:
        unit_iter_train_error.append(variables['avg_train_error'])
        unit_iter_valid_error.append(variables['avg_valid_error'])


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
                                                                         balanced=conf['balanced'], type=conf['type'])

    # if there is not enough data available
    conf['datasets'] = training_data.shape[0]
    if conf['units'] is None:
        conf['units'] = [int(math.ceil((training_data.shape[1] + 7) / 2))]
    conf['n_input'] = training_data.shape[1]
    config = conf

    if conf['unit_range'] is not None:
        del conf['units']
        for units in range(conf['unit_range'][0], conf['unit_range'][1] + 1):
            net = getNet([units], conf['learning_rate'], conf['epochs'], conf['learning_rule'],
                         conf['batch_size'], conf['weight_decay'], conf['dropout_rate'],
                         conf['loss_type'], n_stable=conf['n_stable'], debug=debug, verbose=verbose,
                         callbacks=callbacks,
                         valid_size=conf['ratio']
                         )
            pipeline = getPipeline(training_data, net)
            pipeline.fit(training_data, training_targets)
        utils.plot_lines(data=[unit_iter_train_error, unit_iter_valid_error],
                         labels=["Training error", "Validation error"],
                         xlabel="number of hidden units",
                         ylabel=config['loss_type'],
                         title="training and validation error", suptitle=None, conf=config, additionals=[
                [np.array(unit_iter_train_error).argmin() + conf['unit_range'][0],
                 np.array(unit_iter_train_error).min()],
                [np.array(unit_iter_valid_error).argmin() + conf['unit_range'][0],
                 np.array(unit_iter_valid_error).min()]],
                         begin=conf['unit_range'][0],
                         path="learning/nn/plots/unit_iter/{}_{}.png".format(conf['unit_range'], conf['epochs']))
    else:
        net = getNet(conf['units'], conf['learning_rate'], conf['epochs'], conf['learning_rule'],
                     conf['batch_size'], conf['weight_decay'], conf['dropout_rate'],
                     conf['loss_type'], n_stable=conf['n_stable'], debug=debug, verbose=verbose,
                     callbacks=callbacks,
                     # valid_set=(valid_data, valid_targets)
                     valid_size=conf['ratio']
                     )

        pipeline = getPipeline(training_data, net)
        return pipeline.fit(training_data, training_targets)


def predict(trackName, artistName, track_id, clf):
    data = utils.selectData(track_id)
    y = data.iloc[0]['peak_cat']
    x = data.drop('peak_cat', axis=1)
    res = clf.predict_proba(x)

    print "\nPrediction for {} by {} successfully completed\n\n".format(trackName, artistName)

    match = y == res.argmax()
    print colored("Target category:       {}".format(y), 'green' if match else 'red')
    print colored("Predicted category:    {}".format(res.argmax()), 'green' if match else 'red')

    print "\n\n"
    for cat, prob in enumerate(res[0]):
        if cat == res.argmax() or cat == y:
            if match:
                color = 'green'
            else:
                color = 'red'
        else:
            color = 'blue'
        print colored("Category {}:    {:.3f}%".format(cat, prob * 100), color)
