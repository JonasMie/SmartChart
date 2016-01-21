import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier

import utils


def train(data, target):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, target)
    return clf


def predict(clf, data):
    return clf.predict_proba(data)


def tree_feat_sel(X, y, feature_names, threshold=None, plot=True):
    forest = ExtraTreesClassifier(n_estimators=300,
                                  random_state=0)
    forest.fit(X, y)
    features = []
    all_features = []
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    metadata_std= []
    metadata_imp = []
    metadata_ind = []

    mir_std = []
    mir_imp = []
    mir_ind = []

    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    print("Feature ranking:")
    for f in range(X.shape[1]):
        all_features.append(feature_names[indices[f]])
        if feature_names[indices[f]] in utils.mir:
            mir_std.append(std[indices[f]])
            mir_imp.append(importances[importances[f]])
            mir_ind.append(indices[f])
        else:
            metadata_std.append(std[indices[f]])
            metadata_imp.append(importances[importances[f]])
            metadata_ind.append(indices[f])
        print("{}. feature {} ({})".format(f + 1, feature_names[indices[f]], importances[indices[f]]))
        if threshold and importances[indices[f]] > threshold:
            features.append(feature_names[indices[f]])

    if plot:

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(metadata_ind, metadata_imp,
                color="r", yerr=metadata_std, align="center", width=.5)
        plt.bar(mir_ind, mir_imp,
                color="b", align="center", width=.5)
        plt.xticks(range(X.shape[1]), all_features, rotation='vertical')
        plt.tick_params(axis='x', labelsize=8)
        plt.xlim([-1, X.shape[1]])

        plt.show()

    if threshold:
        return features
    return indices
