import numpy as np
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from learning import learning_utils


def train(conf):
    training_data, training_targets = learning_utils.getData(conf['datasets'], type=conf['type'], split=True,
                                                             balanced=conf['balanced'], shuffle=True)

    clf = tree.DecisionTreeClassifier(criterion=conf['criterion'], splitter='best')
    pipeline = learning_utils.getPipeline(training_data, clf, 'decision_tree')
    return pipeline.fit(training_data, training_targets), training_data.columns.values


def predict(trackName, artistName, track_id, clf):
    learning_utils.predict(trackName, artistName, track_id, clf)


def tree_feat_sel(X, y, feature_names, type, trees=None, threshold=None, plot=True):
    if type == "random":
        clf = RandomForestClassifier(n_estimators=trees)
    elif type == "extra":
        clf = ExtraTreesClassifier(n_estimators=trees)
    elif type == "tree":
        clf = DecisionTreeClassifier()
    clf.fit(X, y)
    features = []
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    ordered_features = []
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("{}. feature {} ({})".format(f + 1, feature_names[indices[f]], importances[indices[f]]))
        # if threshold and importances[indices[f]] > threshold:
        #     features.append(feature_names[indices[f]])
        ordered_features.append(feature_names[indices[f]])
        if threshold and f < threshold * len(importances):
            features.append(feature_names[indices[f]])

    if plot:
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
        # utils.plot_chart(clf.__class__.__name__, trees, importances, indices, ordered_features, std, X.shape[1],type)
        # utils.plot_chart_h(clf.__class__.__name__, trees, importances, indices, ordered_features, std, X.shape[1],type)
        learning_utils.plot_pie(clf.__class__.__name__, trees, importances, indices, ordered_features, threshold,
                                X.shape[1], type)

    if threshold:
        return features
    return indices
