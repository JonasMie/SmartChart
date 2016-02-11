# coding=utf-8
import Tkinter
import tkFileDialog
from argparse import ArgumentParser
from collections import OrderedDict

from mutagen.id3 import ID3
from sklearn.externals import joblib

from MIR.mir import *
from dataCollector import *
from learning import learning_utils
from learning.learning_utils import *
from learning.nn import neuralNetwork
from learning.svm import svc
from learning.tree import decisionTree
from utils import normalizeName


def getTags(path):
    return unicode(ID3(path)["TIT2"].text[0]), unicode(normalizeName(ID3(path)['TPE1'].text[0]))


def getOutput(output):
    if options.output:
        if options.output == "gen":
            return os.path.join(os.getcwd(), 'learning', 'nn', 'models',
                                'units',
                                "{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(options.type, options.units,
                                                                        options.n_iter,
                                                                        options.learning_rate,
                                                                        options.batch_size, options.weight_decay,
                                                                        options.dropout_rate,
                                                                        options.loss_type, int(time())))
        else:
            if os.path.isdir(options.output):
                return os.path.join(options.output,
                                    "{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(options.type, options.units,
                                                                            options.n_iter,
                                                                            options.learning_rate,
                                                                            options.batch_size,
                                                                            options.weight_decay,
                                                                            options.dropout_rate,
                                                                            options.loss_type,
                                                                            int(time())))
            else:
                print options.output + " is not a valid directory"
                sys.exit(2)
    return None


def getPickleFile(pickle_file):
    if pickle_file is None:
        root = Tkinter.Tk()
        root.withdraw()
        file = tkFileDialog.askopenfilename(parent=root, title="Pick a file containing the classifier",
                                            defaultextension="pkl",
                                            filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()
        return joblib.load(file)
    else:
        if os.path.isfile(pickle_file):
            if not input.endswith("pkl"):
                print "Sorry, only pkl-files are supported"
                sys.exit(2)
            else:
                return joblib.load(pickle_file)
        else:
            print "Sorry, the path seems to be incorrect..."
            sys.exit(2)


def parseDirectory(directoryName, extensions):
    '''
    Taken from: 'facerecognitionTemplate' (DataMining)
    This method returns a list of all filenames in the Directory directoryName.
    For each file the complete absolute path is given in a normalized manner (with
    double backslashes). Moreover only files with the specified extension are returned in
    the list.
    '''
    if not os.path.isdir(directoryName): return

    files_found = 0
    artists_found = 0
    files = {}
    for subFolderName in os.listdir(directoryName):
        for root, directories, filenames in os.walk(os.path.join(directoryName, subFolderName)):
            if not root.endswith('.AppleDouble'):
                for filename in filenames:
                    if filename.endswith(
                            extensions) and directories != "":  # and MP3(os.path.join(root, filename)).info.channels == 1:
                        files_found += 1
                        # if files_found == 11:
                        # return files, artists_found, files_found
                        try:
                            path_ = os.path.join(root, filename)
                            trackName, id3ArtistNameNorm = getTags(path_)
                        except KeyError:
                            trackName = unicode(filename.rsplit(".", 1)[0])
                        except:
                            e = sys.exc_info()[0]
                            print colored(u"Fehler:{}".format(e), 'red')
                        if id3ArtistNameNorm not in files:
                            files[id3ArtistNameNorm] = list()
                            artists_found += 1
                        files[id3ArtistNameNorm].append(
                            (unicode(os.path.join(root, filename)), trackName))

    # joblib.dump(files, os.path.join('files', 'new_files.pkl'))
    return files, artists_found, files_found


if __name__ == "__main__":

    total_features = 115
    mir_features = 69
    md_features = 46
    features = None
    gs_params = dict()
    unit_range = None

    parser = ArgumentParser(description="todo")  # todo

    parser.add_argument("-j", "--job",
                        help="Specify what you want to do. Possible values are: train (to train a model), predict "
                             "(to predict the position of a song), collect (to collect data), scores (to do a grid "
                             "search and the according best score), selection(to select the most important features), confusion (plot a confusion matrix)")
    parser.add_argument("-m", "--method",
                        help="Specify the method to use for the given job. Possible values are: net (neural network), "
                             "svm, tree")
    parser.add_argument("-t", "--type", default="all",
                        help="The type of features to use for the job. Possible values are: all (use all features), md "
                             "(only metadata features), mir (only audio features), feat_sel (features retrieved by "
                             "feature selection), random (random amount and types of features)")

    parser.add_argument("-n", "--n-iter", default=200, type=int,
                        help="The number of epochs for neural network training. Default is 200")
    parser.add_argument("-u", "--units", nargs="?", type=int)
    parser.add_argument("-l", "--learning-rate", default=.01, type=float,
                        help="The learning rate for neural network training. Default is 0.01")
    parser.add_argument("-R", "--learning-rule", default="sgd",
                        help="The learning rule for neural network training. Default is 'sgd' (stochastic gradient "
                             "descent)")  # todo : possible values
    parser.add_argument("-b", "--batch-size", default=1, type=int,
                        help="The batch size for neural network training. Default is 1 (online)")
    parser.add_argument("-w", "--weight-decay", default=None,
                        help="The weight decay for neral network training. Default is None")
    parser.add_argument("-e", "--loss-type", default="mcc",
                        help="The loss type for neural network training. Default is 'mcc' (mean categorical "
                             "cross-entropy)")  # todo: possible values
    parser.add_argument("-D", "--dropout-rate", default=None, type=float,
                        help="The dropout rate for neural network training. Default is None")

    parser.add_argument("-s", "--size", default=None, type=int,
                        help="The amount of datasets to use. Default is None (all data is used)")

    parser.add_argument("-o", "--output",  # action="store_true",
                        help="The output path for the model specifications. Use the keyword 'gen' to use th default "
                             "path and a generated filename"
                             "provided, the path is generated.")
    parser.add_argument("-d", "--plot-path",  # action="store_true",
                        help="The output path for the model plots.  Use the keyword 'gen' to use th default path and a "
                             "generated filename")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose output")
    parser.add_argument("-y", "--debug", action="store_true", default=False, help="Enable debugging (neural network)")
    parser.add_argument("-r", "--ratio", default=.2, type=float,
                        help="The ratio used for training/validation split. Default is 0.2")

    parser.add_argument("-S", "--n-stable", default=None, type=int,
                        help="The number of epochs, after which the training is stopped, if the error didn't drop "
                             "significantly")
    parser.add_argument("-B", "--balanced", default=False, action="store_true",
                        help="Set this flag to enable a balanced dataset (same amount of data for each category. "
                             "Warning: This has serious effects on the size of the whole dataset")
    parser.add_argument("-p", "--pickle-file",
                        help="Some jobs require a pickle file to read data from. If path is specified, the "
                             "file chooser will ask you for it.")

    options = parser.parse_args()

    if options.job == "collect" or options.job == "fix":
        if options.pickle_file is not None:
            fileList = joblib.load(options.pickle_file)
            tracks_found = sum(len(y) for y in fileList.itervalues())
        elif input is not None:
            fileList, artists_found, tracks_found = parseDirectory(input, ("mp3"))
        else:
            root = Tkinter.Tk()
            root.withdraw()
            dir = tkFileDialog.askdirectory(parent=root, title='Pick a directory')
            print dir
            root.destroy()
            fileList, artists_found, tracks_found = parseDirectory(dir, ("mp3"))
        if options.job == "collect":
            collectData(fileList, tracks_found)
        elif options.job == "fix":
            fixData(fileList)
    elif options.job == "train":
        if options.method == "net":
            if options.type != 'all' and isinstance(options.type, basestring):
                if options.type == 'mir':
                    i0 = options.mir_features
                elif options.type == 'md':
                    i0 = options.md_features
                elif options.type == 'feat_sel':
                    if options.pickle_file:
                        features = None
                        # get feature list
                        features = joblib.load(options.pickle_file)
                        i0 = len(features)
                    else:
                        print "Please specify  the location of the pickle file (-p) containing the list of features"
                        sys.exit(2)
                elif options.type == 'rand':
                    from utils import features
                    import random

                    features = random.sample(np.hstack(features.values()), random.randint(1, options.total_features))
                    i0 = len(features)
            else:
                i0 = total_features
            if options.units is None and unit_range is None:
                units = [int(math.ceil((i0 + 7) / 2))]

            if options.plot_path:
                if options.plot_path == 'gen':
                    options.plot_path = os.path.join(os.getcwd(), 'learning', 'nn', 'plots',
                                                     'units',
                                                     "{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(options.type,
                                                                                             options.units,
                                                                                             options.n_iter,
                                                                                             options.learning_rate,
                                                                                             options.batch_size,
                                                                                             options.weight_decay,
                                                                                             options.dropout_rate,
                                                                                             options.loss_type,
                                                                                             int(time())))
                else:
                    if os.path.isdir(options.plot_path):
                        options.plot_path = os.path.join(options.plot_path,
                                                         "{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(type, options.units,
                                                                                                 options.n_iter,
                                                                                                 options.learning_rate,
                                                                                                 options.batch_size,
                                                                                                 options.weight_decay,
                                                                                                 options.dropout_rate,
                                                                                                 options.loss_type,
                                                                                                 int(time())))
                    else:
                        print options.plot_path + " is not a valid directory"
                        sys.exit(2)

            parser.output = getOutput(options.output)
            conf = OrderedDict([
                ('datasets', options.size),
                ('type', options.type),
                ('epochs', options.n_iter),
                ('ratio', options.ratio),
                ('units', options.units),
                ('unit_range', unit_range),
                ('n_input', 0),
                ('learning_rate', options.learning_rate),
                ('features', features),
                ('learning_rule', options.learning_rule),
                ('batch_size', options.batch_size),
                ('loss_type', options.loss_type),
                ('weight_decay', options.weight_decay),
                ('dropout_rate', options.dropout_rate),
                ('n_stable', options.n_stable),
                ('balanced', options.balanced)
            ])
            clf = neuralNetwork.train(conf, options.plot_path, gs_params=gs_params, debug=options.debug,
                                      verbose=options.verbose)
            final_attributes = []
            # if options.gs_params:
            #     clf = clf.best_estimator_
            for l in clf._final_estimator.get_parameters():
                final_attributes.append({'layer': l.layer, 'weights': l.weights, 'biases': l.biases})
            clf.final_attributes = final_attributes
            joblib.dump(clf, parser.output, compress=1)
        elif options.method == "tree":
            if size is None:
                size = -1
            if type is None:
                type = "all"
            if options.criterion is None:
                criterion = "gini"
            if options.ratio is None:
                ratio = 1
            if options.output == "":
                output = os.path.join(os.getcwd(), 'learning', 'tree', 'models',
                                      "{}_{}_{}.pkl".format(type, criterion, int(time.time())))
            else:
                if os.path.isdir(options.output):
                    output = os.path.join(options.output,
                                          "{}_{}_{}.pkl".format(type, criterion, int(time.time())))
                else:
                    print options.output + " is not a valid directory"
                    sys.exit(2)

            conf = {
                'datasets': size,
                'type': type,
                'criterion': criterion,
                'balanced': options.balanced
            }
            clf, feature_names = decisionTree.train(conf)
            # joblib.dump(clf, output, compress=1)
            # if plot_path is not None:
            #     if plot_path == "":
            #         plot_path = os.path.join('learning', 'tree', 'plots',
            #                                  "{}_{}_{}.png".format(int(time.time()), size, ratio))
            #     plot(clf._final_estimator, feature_names, config.class_names[0], plot_path)
        elif options.method == "svm":
            if size is None:
                size = -1
            if options.loss_type is None:
                loss_type = "squared_hinge"
            if options.output == "":
                output = os.path.join(os.getcwd(), 'learning', 'svm', 'models',
                                      "{}_{}_{}.pkl".format(size, loss_type, int(time.time())))
            else:
                if os.path.isdir(options.output):
                    output = os.path.join(options.output,
                                          "{}_{}_{}.pkl".format(size, loss_type, int(time.time())))
                else:
                    print options.output + " is not a valid directory"
                    sys.exit(2)

            # balanced = True
            conf = {
                'datasets': size,
                'type': 'all',
                'loss_type': loss_type,
                'ratio': options.ratio,
                'balanced': options.balanced
            }
            clf = svc.train(conf)
            joblib.dump(clf, output, compress=1)
            # if plot_path is not None:
            #     if plot_path == "":
            #         plot_path = os.path.join('learning', 'tree', 'plots',
            #                                  "{}_{}_{}.png".format(int(time.time()), size, ratio))
            #     plot(clf._final_estimator, feature_names, config.class_names[0], plot_path)
    elif options.job == "predict":
        data = dict()
        if type != "rest":
            type = "file"
            if input is None:
                root = Tkinter.Tk()
                root.withdraw()
                input = tkFileDialog.askopenfilename(parent=root, title="Pick a file to predict",
                                                     defaultextension="mp3",
                                                     filetypes=[("Mp3 Files", "*.mp3")])
                root.destroy()
            else:
                if os.path.isfile(input):
                    if not input.endswith("mp3"):
                        print "Sorry, only mp3-files are supported"
                        sys.exit(2)
                else:
                    print "Sorry, the path seems to be incorrect..."
                    sys.exit(2)

            data['trackName'], data['artistName'] = getTags(input)
            data['id'] = collectData({data['artistName']: [(input, data['trackName'])]}, 1, True)

        clf = getPickleFile(options.pickle_file)

        data['type'] = type

        if options.method == "net":
            neuralNetwork.predict(data, clf)
        elif options.method == "tree":
            decisionTree.predict(data, clf)
        elif options.method == "svm":
            svc.predict(data, clf)
    elif options.job == "selection":
        from utils import features

        if type is None or type not in ("tree", "random", "extra"):
            print "Please specify the kind of feature selection model ('-t tree' or '-t random' or '-t extra')"
            sys.exit(2)
        if options.n_iter is None:
            n_iter = 10
        if options.ratio is None:
            ratio = .1
        X, y = getData(size, balanced=False)
        feature_names = X.columns
        X = impute(X)
        features = decisionTree.tree_feat_sel(X, y, feature_names, type, trees=options.n_iter, threshold=options.ratio)
        print features
        if options.output is None:
            output = os.path.join('learning', 'tree', 'features',
                                  "{}_{}_{}_{}.pkl".format(type, size, ratio, int(time.time())))
        joblib.dump(features, output)
    elif options.job == "scores":
        conf = {
            'datasets': options.size,
            'type': options.type,
            'balanced': options.balanced
        }
        if options.method == "net":
            grid_search, training_data, training_targets = neuralNetwork.scores(conf)

            ##wtf ??
            if __name__ == '__main__':
                print("Performing grid search...")
                print("pipeline:", [name for name, _ in grid_search.estimator.steps])
                print("parameters:")
                pprint(grid_search.param_grid)
                t0 = time()
                grid_search.fit(training_data, training_targets)
                print("done in %0.3fs" % (time() - t0))
                print()

                print("Best score: %0.3f" % grid_search.best_score_)
                print("Best parameters set:")
                best_parameters = grid_search.best_estimator_.get_params()
                for param_name in sorted(grid_search.param_grid.keys()):
                    print("\t%s: %r" % (param_name, best_parameters[param_name]))
        elif options.method == "tree":
            tree_type = "extra"  # todo
            if tree_type is None:
                print "You must specify the type of tree (-T tree or -T random or -T extra)"
                sys.exit(2)
            conf['tree'] = tree_type
            decisionTree.scores(conf)
        elif options.method == "svm":
            svc.scores(conf)
    elif options.job == "grid_search":
        conf = OrderedDict([
            ('datasets', size),
            ('epochs', (10, 20))
        ])
        clf = neuralNetwork.train(conf, options.plot_path, debug=options.debug, verbose=options.verbose)
        final_attributes = []
        for l in clf._final_estimator.get_parameters():
            final_attributes.append({'layer': l.layer, 'weights': l.weights, 'biases': l.biases})
        clf.final_attributes = final_attributes
        joblib.dump(clf, options.output, compress=1)
    elif options.job == "confusion":
        clf = getPickleFile(options.pickle_file)
        learning_utils.plot_confusion(clf, balanced=options.balanced)
    elif options.job == "histogram":
        conf = {
            'datasets': options.size,
            'type': options.type,
            'balanced': options.balanced,
            'ratio': .2,
            'features': None,
            'epochs': options.n_iter,
            'unit_range': None
        }
        neuralNetwork.hist(2, conf)
    else:
        print ("No job provided.")
        parser.print_help()
