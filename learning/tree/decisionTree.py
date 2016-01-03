from sklearn import tree


def train(data, target):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, target)
    return clf


def predict(clf, data):
    return clf.predict_proba(data)
