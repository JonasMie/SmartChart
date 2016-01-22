# coding=utf-8
import Tkinter
import getopt
import tkFileDialog

from mutagen.id3 import ID3
from sklearn.externals import joblib

import config
from MIR.mir import *
from dataCollector import *
from learning.nn import neuralNetwork
from learning.tree import decisionTree
from learning.utils import *
from utils import normalizeName


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
                            print path_
                            trackName = unicode(ID3(path_)["TIT2"].text[0])
                            id3ArtistName = unicode(ID3(path_)['TPE1'].text[0])
                            id3ArtistNameNorm = unicode(normalizeName(id3ArtistName))
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

    job = 'collect'
    method = 'net'
    size = None
    output = None
    ratio = None
    plot_path = None

    input_dir = None
    pickle_file = None

    units = None
    learning_rate = None
    n_iter = None
    learning_rule = None,
    batch_size = None
    weight_decay = None
    dropout_rate = None
    loss_type = None
    debug = False
    verbose = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:j:i:p:m:o:s:r:d:u:l:n:b:w:e:y:D:R:v",
                                   ["help", "job=", "input=", "pickle=", "method=", "output=",
                                    "size=", "ratio=", "draw=", "units=", "learningrate=", "iterations=", "batchsize=",
                                    "weightdecay=", "errortype=", "debug=", "dropoutrate=", "learningrule=",
                                    "verbose="])
    except:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-j", "--job"):
            job = a
        elif o in ("-i", "--input"):
            input_dir = a
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
            units = int(a)
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
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    if job == "collect" or job == "fix":
        if pickle_file is not None:
            fileList = joblib.load(pickle_file)
            tracks_found = sum(len(y) for y in fileList.itervalues())
        elif input_dir is not None:
            fileList, artists_found, tracks_found = parseDirectory(input_dir, ("mp3"))
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

            features = None
            # get feature list
            if pickle_file is not None:
                features = joblib.load(pickle_file)

            if size is None:
                size = 5000
            if ratio is None:
                ratio = .8
            if learning_rate is None:
                learning_rate = .01
            if learning_rule is None:
                learning_rule = 'sgd'
            if batch_size is None:
                batch_size = 1  # online
            if weight_decay is None:
                weight_decay = None
            if loss_type is None:
                loss_type = 'mcc'
            if n_iter is None:
                n_iter = 100
            if units is None:
                if features:
                    units = math.ceil((len(features)+7)/2)


            neuralNetwork.train(size=size, ratio=ratio, units=units, learning_rate=learning_rate, iterations=n_iter,
                                features=features, learning_rule=learning_rule, batch_size=batch_size,
                                weight_decay=weight_decay, dropout_rate=dropout_rate, loss_type=loss_type, debug=debug,
                                verbose=verbose)


            # if output is None:
            #     output = os.path.join(os.getcwd(), 'learning', 'nn', 'models',
            #                           "{}_{}_{}.pkl".format(int(time.time())))
            # else:
            #     if os.path.isdir(output):
            #         output = os.path.join(output, "{}_{}_{}.pkl".format(size, ratio, time.time()))
            #     else:
            #         print output + " is not a valid directory"
            #         sys.exit(2)
            # joblib.dump(clf, output)

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
            pass
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
            data = getPredictionData(3081)
            print decisionTree.predict(clf, data)
    elif job == "selection":
        if size is None:
            size = 6000
        X, y = getData(size, balanced=True)
        feature_names = X.columns
        X = impute(X)
        features = decisionTree.tree_feat_sel(X, y, feature_names, threshold=0.01)
        print features
        if output is not None:
            joblib.dump(features, output)

    elif job == "test":
        check1(fileList=joblib.load("files/new_files.pkl"))
