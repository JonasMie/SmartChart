from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from learning import learning_utils


def train(conf):
    training_data, training_targets, ids = learning_utils.getData(conf['datasets'], ratio=conf['ratio'],
                                                                  type=conf['type'], split=True,
                                                                  balanced=conf['balanced'], shuffle=True,
                                                                  return_ids=True)

    training_data, test_data, training_targets, test_targets = cross_validation.train_test_split(training_data,
                                                                                                 training_targets,
                                                                                                 test_size=0.4,
                                                                                                 random_state=0)

    parameters = [
        {
            'svm__kernel': ['rbf'],
            # 'svm__gamma': [1e-3, 1e-4],
            'svm__C': [1, 10, 100, 1000]
        },
        {
            'svm__kernel': ['linear'],
            'svm__C': [1, 10, 100, 1000]
        }
    ]

    clf_ = svm.SVC()
    pipeline = learning_utils.getPipeline(training_data, clf_, 'svm')
    # grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(pipeline, parameters, cv=5, verbose=1,
                           scoring='%s_weighted' % score)
        clf.fit(training_data, training_targets)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_targets, clf.predict(test_data)
        print(classification_report(y_true, y_pred))
        print()

        # clf = svm.SVC(C=.9)  # loss=conf['loss_type'])
        # clf1 = svm.LinearSVC()
        # clf.ids = ids
        # pipeline = learning_utils.getPipeline(training_data, clf, 'svm')
        # pipeline1 = learning_utils.getPipeline(training_data, clf1, 'svm')
        #
        # scores = cross_validation.cross_val_score(pipeline, training_data, training_targets, cv=5)
        # scores1 = cross_validation.cross_val_score(pipeline1, training_data, training_targets, cv=5)
        # print scores
        # print scores1
        # model = pipeline.fit(training_data, training_targets)


        # return scores  # model, training_data.columns.values


def predict(data, clf):
    learning_utils.predict(data, clf[0], False)
