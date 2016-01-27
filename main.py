# coding=utf-8
import Tkinter
import getopt
import tkFileDialog
from collections import OrderedDict

from mutagen.id3 import ID3
from sklearn.externals import joblib

import config
from MIR.mir import *
from dataCollector import *
from learning.nn import neuralNetwork
from learning.tree import decisionTree
from learning.utils import *
from utils import normalizeName


def getTags(path):
    return unicode(ID3(path)["TIT2"].text[0]), unicode(normalizeName(ID3(path)['TPE1'].text[0]))


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


def usage():
    print "Available options:"
    print "\tjob:string (the task to perform (one of collect,...))"
    print "\tpickle:string (the pickle file with saved track paths)"
    print "\tinput:string (input directory containing the files to analyze)"


if __name__ == "__main__":

    total_features = 115
    mir_features = 69
    md_features = 46
    job = 'collect'
    method = 'net'
    size = None
    output = None
    ratio = None
    plot_path = None
    type = 'all'
    input = None
    pickle_file = None
    features = None
    layers = None
    units = None
    learning_rate = None
    n_iter = None
    learning_rule = None
    batch_size = None
    weight_decay = None
    dropout_rate = None
    loss_type = None
    n_stable = None
    debug = False
    verbose = None
    balanced = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:j:i:t:p:m:o:s:r:d:u:l:n:b:w:e:y:v:D:R:S:B",
                                   ["help", "job=", "input=", "pickle=", "method=", "output=",
                                    "size=", "ratio=", "draw=", "units=", "learningrate=", "iterations=",
                                    "batchsize=",
                                    "weightdecay=", "errortype=", "debug=", "verbose=", "dropoutrate=", "learningrule=",
                                    "n_stable=", "balanced="
                                    ])
    except:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-j", "--job"):
            job = a
        elif o in ("-i", "--input"):
            input = a
        elif o in ("-t", "--type"):
            type = a
        elif o in ("-p", "--pickle"):
            pickle_file = a
        elif o in ("-m", "--method"):
            method = a
        elif o in ("-o", "--output"):
            output = a
        elif o in ("-s", "--size"):
            size = int(a)
        elif o in ("-r", "--ratio"):
            ratio = float(a)
        elif o in ("-d", "--draw"):
            plot_path = a
        elif o in ("-u", "--units"):
            units = [int(unit) for unit in a.split()]
        elif o in ("-l", "--learningrate"):
            learning_rate = float(a)
        elif o in ("-n", "--n_iterations"):
            n_iter = int(a)
        elif o in ("-w", "--weightdecay"):
            weight_decay = float(a)
        elif o in ("-b", "--batchsize"):
            batch_size = int(a)
        elif o in ("-e", "--errortype"):
            loss_type = a
        elif o in ("-D", "--dropoutrate"):
            dropout_rate = a
        elif o in ("-R", "--learningrule"):
            learning_rule = a
        elif o in ("-y", "--debug"):
            debug = a
        elif o in ("-v", "--verbose"):
            verbose = a
        elif o in ("-S", "--stable"):
            n_stable = int(a)
        elif o in ("-B", "--balanced"):
            balanced = bool(a)
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    if job == "collect" or job == "fix":
        if pickle_file is not None:
            fileList = joblib.load(pickle_file)
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
        if job == "collect":
            collectData(fileList, tracks_found)
        elif job == "fix":
            fixData(fileList)
    elif job == "train":
        if method == "net":
            if ratio is None:
                ratio = .2
            if learning_rate is None:
                learning_rate = .01
            if learning_rule is None:
                learning_rule = 'sgd'
            if batch_size is None:
                batch_size = 1  # online
            if weight_decay is None:
                weight_decay = None
            if loss_type is None:
                loss_type = 'mse'
            if n_iter is None:
                n_iter = 1000
            if type != 'all':
                if type == 'mir':
                    i0 = mir_features
                elif type == 'md':
                    i0 = md_features
                elif type == 'feat_sel':
                    if pickle_file:
                        features = None
                        # get feature list
                        features = joblib.load(pickle_file)
                        i0 = len(features)
                    else:
                        print "Please specify  the location of the pickle file (-p) containing the list of features"
                        sys.exit(2)
                elif type == 'rand':
                    from utils import features
                    import random

                    features = random.sample(np.hstack(features.values()), random.randint(1, total_features))
                    i0 = len(features)
            else:
                i0 = total_features
            if units is None:
                units = [int(math.ceil((i0 + 7) / 2))]

            if plot_path is not None:
                if plot_path == "":
                    plot_path = os.path.join(os.getcwd(), 'learning', 'nn', 'plots',
                                             'units',
                                             "{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(type, units, n_iter,
                                                                                     learning_rate,
                                                                                     batch_size, weight_decay,
                                                                                     dropout_rate,
                                                                                     loss_type, int(time.time())))
                else:
                    if os.path.isdir(plot_path):
                        output = os.path.join(plot_path,
                                              "{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(type, units, n_iter,
                                                                                      learning_rate,
                                                                                      batch_size, weight_decay,
                                                                                      dropout_rate, loss_type,
                                                                                      int(time.time())))
                    else:
                        print plot_path + " is not a valid directory"
                        sys.exit(2)

            if output == "":
                output = os.path.join(os.getcwd(), 'learning', 'nn', 'models',
                                      'units',
                                      "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(type, type, units, n_iter,
                                                                                 learning_rate,
                                                                                 batch_size, weight_decay, dropout_rate,
                                                                                 loss_type, int(time.time())))
            else:
                if os.path.isdir(output):
                    output = os.path.join(output,
                                          "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(type, type, units, n_iter,
                                                                                     learning_rate,
                                                                                     batch_size, weight_decay,
                                                                                     dropout_rate, loss_type,
                                                                                     int(time.time())))
                else:
                    print output + " is not a valid directory"
                    sys.exit(2)

            conf = OrderedDict([
                ('datasets', size),
                ('type', type),
                ('epochs', n_iter),
                ('ratio', ratio),
                ('units', units),
                ('n_input', 0),
                ('learning_rate', learning_rate),
                ('features', features),
                ('learning_rule', learning_rule),
                ('batch_size', batch_size),
                ('loss_type', loss_type),
                ('weight_decay', weight_decay),
                ('dropout_rate', dropout_rate),
                ('n_stable', n_stable),
                ('balanced', balanced)
            ])
            clf = neuralNetwork.train(conf, plot_path, debug=debug, verbose=verbose)

            joblib.dump(clf, output, compress=1)

        elif method == "tree":
            if size is None:
                size = 200
            if ratio is None:
                ratio = 1
            if output is None:
                output = os.path.join(os.getcwd(), 'learning', 'tree', 'models',
                                      "{}_{}_{}.pkl".format(int(time.time()), size, ratio))
            else:
                if os.path.isdir(output):
                    output = os.path.join(output, "{}_{}_{}.pkl".format(size, ratio, time.time()))
                else:
                    print output + " is not a valid directory"
                    sys.exit(2)
            data, targets = getDecisionData(size, ratio)
            feature_names = data.columns

            data = impute(data)
            clf = decisionTree.train(data, targets.values)
            joblib.dump(clf, output)
            if plot_path is not None:
                if plot_path == "":
                    plot_path = os.path.join('learning', 'tree', 'plots',
                                             "{}_{}_{}.png".format(int(time.time()), size, ratio))
                plot(clf, feature_names, config.class_names[0], plot_path)
    elif job == "predict":
        if method == "net":
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

            trackName, artistName = getTags(input)

            if pickle_file is None:
                root = Tkinter.Tk()
                root.withdraw()
                pickle_file = tkFileDialog.askopenfilename(parent=root, title="Pick a file containing the classifier",
                                                           defaultextension="pkl",
                                                           filetypes=[("Pickle Files", "*.pkl")])
                root.destroy()
            else:
                if os.path.isfile(pickle_file):
                    if not input.endswith("pkl"):
                        print "Sorry, only pkl-files are supported"
                        sys.exit(2)
                else:
                    print "Sorry, the path seems to be incorrect..."
                    sys.exit(2)
            clf = joblib.load(pickle_file)
            neuralNetwork.predict(trackName, artistName, collectData({artistName: [(input, trackName)]}, 1, True), clf)


        elif method == "tree":
            if pickle_file is None:
                files = os.listdir(os.path.join('learning', 'tree', 'models'))
                i = len(files) - 1
                while i >= 0:
                    if files[i].endswith('.pkl'):
                        pickle_file = os.path.join('learning', 'tree', 'models', files[i])
                        break
                    i -= 1
            if pickle_file is None:
                print "You must specify a pickle file containing the trained model"
                sys.exit(2)
            clf = joblib.load(pickle_file)
            data = selectData(3081)
            print decisionTree.predict(clf, data)
    elif job == "selection":
        if size is None:
            size = 8000
        X, y = getData(size, balanced=True)
        feature_names = X.columns
        X = impute(X)
        features = decisionTree.tree_feat_sel(X, y, feature_names, threshold=0.01)
        print features
        if output is not None:
            joblib.dump(features, output)

    elif job == "test":
        check1(fileList=joblib.load("files/new_files.pkl"))
