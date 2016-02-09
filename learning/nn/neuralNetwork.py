from __future__ import division

import math

import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sknn.mlp import Classifier, Layer

from learning import learning_utils

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

unit_iter_best_train_error = []
unit_iter_best_valid_error = []


def on_train_start(**variables):
    pass


def on_train_finish(**variables):
    global plot
    if plot is not None:
        learning_utils.plot_lines(data=[train_errors, valid_errors], labels=["Training error", "Validation error"],
                                  xlabel="number of epochs",
                                  ylabel=config['loss_type'],
                                  title="training and validation error", suptitle=None, conf=config, additionals=[
                [best_train_i, variables['best_train_error']], [best_valid_i, variables['best_valid_error']]],
                                  path=plot)
    from sklearn.externals import joblib
    joblib.dump(valid_errors, 'test.pkl', compress=1)
    if config['unit_range']:
        unit_iter_train_error.append(variables['avg_train_error'])
        unit_iter_valid_error.append(variables['avg_valid_error'])

        unit_iter_best_train_error.append(variables['best_train_error'])
        unit_iter_best_valid_error.append(variables['best_valid_error'])


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
    'on_epoch_finish': on_epoch_finish,
    'on_train_finish': on_train_finish
}


def getNet(units=[61], learning_rate=0.01, n_iter=16, learning_rule='sgd', batch_size=1, weight_decay=None,
           dropout_rate=None, loss_type='mcc', n_stable=10,
           debug=False, verbose=False, callbacks=None, valid_size=.2):
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
        valid_size=valid_size,
        regularize='L2'
    )


def getData(size, ratio, features, balanced, type):
    complete_data = learning_utils.getData(size, split=False, balanced=balanced, type=type)

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


def train(conf, plot_path, debug, verbose, gs_params=None, callbacks=default_callbacks):
    global config, plot, train_errors, valid_errors
    plot = plot_path
    if 'neural_network__n_iter' in gs_params:
        train_errors = np.zeros(20)
        valid_errors = np.zeros(20)
    else:
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
            pipeline = learning_utils.getPipeline(training_data, net, 'neural_network')
            pipeline.fit(training_data, training_targets)
        learning_utils.plot_lines(data=[unit_iter_train_error, unit_iter_valid_error],
                                  labels=["Training error", "Validation error"],
                                  xlabel="number of hidden units",
                                  ylabel=config['loss_type'],
                                  title="training and validation error", suptitle=None, conf=config, additionals=[
                [np.array(unit_iter_train_error).argmin() + conf['unit_range'][0],
                 np.array(unit_iter_train_error).min()],
                [np.array(unit_iter_valid_error).argmin() + conf['unit_range'][0],
                 np.array(unit_iter_valid_error).min()]],
                                  begin=conf['unit_range'][0],
                                  path="learning/nn/plots/unit_iter/{}_{}.png".format(conf['unit_range'],
                                                                                      conf['epochs']))
        learning_utils.plot_lines(data=[unit_iter_best_train_error, unit_iter_best_valid_error],
                                  labels=["Training error", "Validation error"],
                                  xlabel="number of hidden units",
                                  ylabel=config['loss_type'],
                                  title="training and validation error", suptitle=None, conf=config, additionals=[
                [np.array(unit_iter_best_train_error).argmin() + conf['unit_range'][0],
                 np.array(unit_iter_best_train_error).min()],
                [np.array(unit_iter_best_valid_error).argmin() + conf['unit_range'][0],
                 np.array(unit_iter_best_valid_error).min()]],
                                  begin=conf['unit_range'][0],
                                  path="learning/nn/plots/unit_iter/{}_{}_{}.png".format(conf['unit_range'],
                                                                                         conf['epochs'],
                                                                                         "best"))
    else:
        net = getNet(conf['units'], conf['learning_rate'], conf['epochs'], conf['learning_rule'],
                     conf['batch_size'], conf['weight_decay'], conf['dropout_rate'],
                     conf['loss_type'], n_stable=conf['n_stable'], debug=debug, verbose=verbose,
                     callbacks=callbacks,
                     # valid_set=(valid_data, valid_targets)
                     valid_size=conf['ratio']
                     )
        pipeline = learning_utils.getPipeline(training_data, net, 'neural_network')

        if gs_params:
            gs = GridSearchCV(pipeline, param_grid=gs_params)
            return gs.fit(training_data, training_targets)

        return pipeline.fit(training_data, training_targets)


def predict(trackName, artistName, track_id, clf):
    learning_utils.predict(trackName, artistName, track_id, clf)


def train_custom(conf, plot_path, debug, verbose, gs_params=None, callbacks=default_callbacks):
    global config, plot, train_errors, valid_errors
    plot = plot_path

    train_errors = np.zeros(conf['epochs'])
    valid_errors = np.zeros(conf['epochs'])
    all_ = list()
    for x in ['all', 'md', 'mir', 'feat_sel', 'random']:
        all = list()
        if x in ('all', 'md', 'mir'):
            training_data, training_targets, valid_data, valid_targets = getData(conf['datasets'], 0, None,
                                                                                 None, type=x)
        elif x == 'random':
            from utils import features
            import random
            conf['features'] = random.sample(np.hstack(features.values()), random.randint(1, 115))
            training_data, training_targets, valid_data, valid_targets = getData(conf['datasets'], 0, conf['features'],
                                                                                 balanced=conf['balanced'], type=x)
        else:
            training_data, training_targets, valid_data, valid_targets = getData(conf['datasets'], 0, conf['features'],
                                                                                 balanced=conf['balanced'], type=x)

        # if there is not enough data available
        conf['datasets'] = training_data.shape[0]
        if conf['units'] is None:
            conf['units'] = [int(math.ceil((training_data.shape[1] + 7) / 2))]
        conf['n_input'] = training_data.shape[1]
        config = conf
        units = [int(math.ceil((training_data.shape[1] + 7) / 2))]
        for i in range(1, 101):
            net = getNet(units, conf['learning_rate'], conf['epochs'], conf['learning_rule'],
                         conf['batch_size'], conf['weight_decay'], conf['dropout_rate'],
                         conf['loss_type'], n_stable=conf['n_stable'], debug=debug, verbose=verbose,
                         callbacks=callbacks,
                         # valid_set=(valid_data, valid_targets)
                         valid_size=conf['ratio']
                         )
            pipeline = learning_utils.getPipeline(training_data, net)
            pipeline.fit(training_data, training_targets)
            all.append(valid_errors)
        all_.append(np.array(all).mean(axis=0))
    learning_utils.plot_lines(data=all_, labels=["all", "md", "mir", "feat_sel", "random"],
                              xlabel="number of epochs",
                              ylabel=config['loss_type'],
                              title="mean training and validation error", suptitle=None,
                              path="learning/nn/plots/comb/test.png")


def scores(conf):
    parameters = {
        'nn__n_iter': range(10, 25),
        'nn__layers': [
            # [Layer(type="Softmax", name="output")],
            [Layer(type="Sigmoid", units=10, name="h0"), Layer(type="Softmax", name="output")],
            [Layer(type="Sigmoid", units=20, name="h0"), Layer(type="Softmax", name="output")],
            [Layer(type="Sigmoid", units=35, name="h0"), Layer(type="Softmax", name="output")],
            [Layer(type="Sigmoid", units=50, name="h0"), Layer(type="Softmax", name="output")],
            [Layer(type="Sigmoid", units=75, name="h0"), Layer(type="Softmax", name="output")],
            # [Layer(type="Sigmoid", units=50, name="h0"), Layer(type="Sigmoid", units=50, name="h1"),
             # Layer(type="Softmax", name="output")]
        ],
        'nn__learning_rate': [.001, .01, .05],
        # 'nn__learning_momentum': [.8, .9],
        'nn__batch_size': [1, 10, 25, 50],
        'nn__dropout_rate': [0, .1, .25, .5]

    }
    training_data, training_targets = learning_utils.getData(conf['datasets'], type=conf['type'], split=True,
                                                             balanced=conf['balanced'], shuffle=True)
    clf = getNet(valid_size=0)
    pipeline = learning_utils.getPipeline(training_data, clf, 'nn')
    grid_search = GridSearchCV(pipeline, parameters,verbose=10)
    return grid_search, training_data, training_targets
    # learning_utils.gs(grid_search, training_data, training_targets)
