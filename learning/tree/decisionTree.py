import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def train(data, target):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, target)
    return clf


def predict(clf, data):
    return clf.predict_proba(data)


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
        # Plot the feature importances of the forest
        plt.figure(figsize=(20, 6))
        plt.title("Feature importances according to the {}  (estimators: {})".format(clf.__class__.__name__, trees))
        # plt.("estimators: {}".format(trees))
        if type in ("random", "extra"):
            std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                         axis=0)
            plt.bar(range(X.shape[1]), importances[indices],
                    color="r", yerr=std[indices], align="center")
        else:
            plt.bar(range(X.shape[1]), importances[indices],
                    color="r", align="center")

        plt.xticks(range(X.shape[1]), ordered_features, rotation='vertical', fontsize=8)
        plt.xlim([-1, X.shape[1]])
        plt.ylim([-.01,.1])
        plt.show()

    if threshold:
        return features
    return indices
