# coding=utf-8
import Tkinter, tkFileDialog
from os import listdir
from os.path import *

from mutagen.id3 import ID3
from mutagen.mp3 import MP3

from MIR.mir import *
from dataCollector import collectData
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
    if not isdir(directoryName): return

    folders = list()

    files_found = 0
    artists_found = 0
    files = {}
    for subFolderName in listdir(directoryName):
        for root, directories, filenames in os.walk(os.path.join(directoryName, subFolderName)):
            for filename in filenames:
                if filename.endswith(extensions):  # and MP3(os.path.join(root, filename)).info.channels == 1:
                    files_found += 1
                    try:
                        trackName = ID3(os.path.join(root, filename))["TIT2"].text[0]
                        id3ArtistName = ID3(os.path.join(root, filename))['TPE1'].text[0]
                        id3ArtistNameNorm = normalizeName(id3ArtistName)
                    except KeyError:
                        trackName = filename.rsplit(".", 1)[0].encode('utf-8')
                    if id3ArtistNameNorm not in files:
                        files[id3ArtistNameNorm] = list()
                        artists_found += 1
                    files[id3ArtistNameNorm].append(
                            (os.path.join(root, filename).encode('utf-8'), trackName))
                    # folders.append((subFolderName.encode('utf-8'), subFolderFiles))

    return files, artists_found, files_found


def usage():
    print "Available options:"
    print "\tinput:string (input file .wav)"
    print "\touput:string (output file .png)"
    print "\tmethod:string (valid methods: correlogram, spectrogram, amdfogram)"
    print "\tplot_title:string"
    print "\twin_size:int"


if __name__ == "__main__":
    root = Tkinter.Tk()
    root.withdraw()

    dir = tkFileDialog.askdirectory(parent=root, title='Pick a directory')
    root.destroy()
    # dir = tkFileDialog.askdirectory(parent=root)

    fileList, artists_found, tracks_found = parseDirectory(dir, ("mp3"))
    print "Found {} files ({} artists)".format(tracks_found, artists_found)

    collectData(fileList)
