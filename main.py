# coding=utf-8
import Tkinter
import getopt
import tkFileDialog

from mutagen.id3 import ID3
from sklearn.externals import joblib

import config
from MIR.mir import *
from dataCollector import *
from learning.tree import decisionTree
from learning.utils import *
from utils import normalizeName


# '''
# Request the detailed track page and search for the track's peak position
# '''
# detail = requests.get("https://www.offiziellecharts.de%s" % track_url)
# parsed_detail = BeautifulSoup(detail.text, "html.parser")
# try:
#     table_row = parsed_detail.find('table', class_='chart-table').findChildren()[6]
# except AttributeError:
#     return None
#
# '''
# Check if the found table row contains the peak position
# '''
# if table_row.findChildren()[0].string == unicode("Höchstposition:", encoding='utf-8'):
#     peak_re = re.search("[^\s]+", table_row.findChildren()[1].string)
#     if peak_re:
#         return peak_re.group()
# return None


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

    joblib.dump(files, os.path.join('files', 'new_files.pkl'))
    return files, artists_found, files_found


def usage():
    print "Available options:"
    print "\tjob:string (the task to perform (one of collect,...))"
    print "\tpickle:string (the pickle file with saved track paths)"
    print "\tinput:string (input directory containing the files to analyze)"


if __name__ == "__main__":

    job = 'collect'
    method = 'net'
    size = 200
    output = None
    ratio = 1
    plot_path = None

    input_dir = None
    pickle_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:j:i:p:m:o:s:r:d",
                                   ["help", "job=", "input=", "pickle=", "method=", "output=",
                                    "size=", "ratio=", "draw="])
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
            output_file = a
        elif o in ("-s", "--size"):
            size = a
        elif o in ("-r", "--ratio"):
            ratio = a
        elif o in ("-d", "--draw"):
            plot_path = a
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
            if output is None:
                output = os.path.join(os.getcwd(), 'learning', 'nn', 'models',
                                      "{}_{}_{}.pkl".format(int(time.time())))
            else:
                if os.path.isdir(output):
                    output = os.path.join(output, "{}_{}_{}.pkl".format(size, ratio, time.time()))
                else:
                    print output + " is not a valid directory"
                    sys.exit(2)
            data, targets = getDecisionData(size, ratio)
            feature_names = data.columns

            # clf = neuralNetwork.getPipeline().fit() #data, targets.values
            joblib.dump(clf, output)
        elif method == "tree":
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
        X, y = getData(300)
        feature_names = X.columns
        X = impute(X)
        decisionTree.tree_feat_sel(X, y, feature_names)
    elif job =="test":
        check1( fileList = joblib.load("files/new_files.pkl"))