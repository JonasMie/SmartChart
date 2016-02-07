import numpy as np
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV

from learning import learning_utils


def train(conf):
    training_data, training_targets, ids = learning_utils.getData(conf['datasets'], ratio=conf['ratio'],
                                                                  type=conf['type'], split=True,
                                                                  balanced=conf['balanced'], shuffle=True,
                                                                  return_ids=True)
    clf = svm.SVC(C=1, kernel='rbf')
    clf1 = svm.SVC(C=100, kernel='rbf')  # loss=conf['loss_type'])
    # clf1 = svm.LinearSVC(C=.9)
    clf.ids = ids
    pipeline = learning_utils.getPipeline(training_data, clf, 'svm')
    pipeline1 = learning_utils.getPipeline(training_data, clf1, 'svm')

    scores = cross_validation.cross_val_score(pipeline, training_data, training_targets, cv=5, verbose=True, n_jobs=-1)
    scores1 = cross_validation.cross_val_score(pipeline1, training_data, training_targets, cv=5, verbose=True,
                                               n_jobs=-1)
    print scores
    print scores1

    return  pipeline.fit(training_data, training_targets), training_data.columns.values


def predict(data, clf):
    learning_utils.predict(data, clf[0], False)


def scores(conf):
    parameters = {
        'svm__C': np.logspace(-2, 5, 8),
        'svm_gamma': np.logspace(-9, 2, 12),
        'svm__kernel': ['linear', 'rbf']
    }
    training_data, training_targets = learning_utils.getData(conf['datasets'], type=conf['type'], split=True,
                                                             balanced=conf['balanced'], shuffle=True)
    clf = svm.SVC()
    pipeline = learning_utils.getPipeline(training_data, clf, 'nn')
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    learning_utils.gs(grid_search, training_data, training_targets)
