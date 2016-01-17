import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier


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
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("{}. feature {} ({})".format(f + 1, feature_names[indices[f]], importances[indices[f]]))
        if threshold and importances[indices[f]]> threshold:
            features.append(feature_names[indices[f]])

    if plot:
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()

    if threshold:
        return features
    return indices
