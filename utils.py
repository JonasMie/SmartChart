import datetime
import difflib
import re
import sys

from dateutil import parser

feat = re.compile(r"([\[(](?:ft?\.|featuring|feat(?:[\.]|))(.*)[\])])", re.I)
progress_x = 0

mir = (
'zcr', 'zcr_std', 'nrg', 'nrg_std', 'pow', 'pow_std', 'acr', 'acr_std', 'cent', 'cent_std', 'flx', 'flx_std', 'rlf',
'rlf_std', 'mfcc_0', 'mfcc_0_std', 'mfcc_1', 'mfcc_1_std', 'mfcc_2', 'mfcc_2_std', 'mfcc_3', 'mfcc_3_std', 'mfcc_4',
'mfcc_4_std', 'mfcc_5', 'mfcc_5_std', 'mfcc_6', 'mfcc_6_std', 'mfcc_7', 'mfcc_7_std', 'mfcc_8', 'mfcc_8_std', 'mfcc_9',
'mfcc_9_std', 'mfcc_10', 'mfcc_10_std', 'mfcc_11', 'mfcc_11_std', 'mfcc_12', 'mfcc_12_std', 'chr_0', 'chr_0_std',
'chr_1', 'chr_1_std', 'chr_2', 'chr_2_std', 'chr_3', 'chr_3_std', 'chr_4', 'chr_4_std', 'chr_5', 'chr_5_std', 'chr_6',
'chr_6_std', 'chr_7', 'chr_7_std', 'chr_8', 'chr_8_std', 'chr_9', 'chr_9_std', 'chr_10', 'chr_10_std', 'chr_11',
'chr_11_std', 'eoe', 'eoe_std', 'eoe_min')
metadata_artist = (
'is_male', 'is_female', 'is_group', 'german', 'american', 'other_country', 'total_years', 'life_span',
'artist_genre_electronic', 'artist_genre_pop', 'artist_genre_hiphop', 'artist_genre_rock', 'artist_genre_other',
'followers', 'listener', 'play_count', 'popularity', 'mean_chart_peak_0', 'mean_chart_peak_1', 'mean_chart_peak_2',
'mean_chart_peak_3', 'mean_chart_peak_4', 'mean_chart_peak_5', 'mean_chart_peak_6', 'mean_chart_weeks',
'total_chart_weeks', 'mean_album_chart_peak_0', 'mean_album_chart_peak_1', 'mean_album_chart_peak_2',
'mean_album_chart_peak_3', 'mean_album_chart_peak_4', 'mean_album_chart_peak_5', 'mean_album_chart_peak_6',
'mean_album_chart_weeks', 'total_album_chart_weeks')
metadata_track = (
'track_genre_electronic', 'track_genre_pop', 'track_genre_hiphop', 'track_genre_rock', 'track_genre_other', 'is_1980s',
'is_1990', 'is_2000', 'is_2010s', 'is_other_decade', 'length')


def getActivity(start, end=datetime.datetime.now(), format="%Y-%m-%d"):
    if isinstance(start, basestring):
        # start = datetime.datetime.strptime(start, format)
        start = parser.parse(start)
    if isinstance(end, basestring):
        end = parser.parse(end)
    return (end - start).days


def getCurrentYear():
    return datetime.datetime.now().year


def checkTrackNamingConvention(actual, target, normalize=False):
    variations = ["", " Radio Edit", " (Radio Edit)", " - Radio Edit", " -Radio Edit"]
    for variation in variations:
        if is_similar(actual, target + variation, normalize):
            return True
    return False


def isTrackRemixByName(track):
    return "Remix" in track


def getDuration(length):
    duration = 0
    i = 0
    time_entities = [int(n) for n in length.split(":")]
    for val in reversed(time_entities):
        duration += int(val) * 60 ** i
        i += 1
    return duration


def normalizeName(track):
    '''

    :param track:
    :return: clean track name
    '''

    '''
    Remove any 'featuring' from trackname
    e.g. 'feat. XY', 'ft. XY', 'featuring XY',...
    '''
    return re.sub(r"([\[(](\s)*(?:ft?\.|featuring|feat(?:[\.]|))(.*)[\])])", '',
                  track.lower()).strip()  # TODO: nur bis zur ersten klammer (id: 2827)


def is_similar(name1, name2, normalize=False, border=.9):
    if normalize:
        name1 = normalizeName(name1)
        name2 = normalizeName(name2)
    return difflib.SequenceMatcher(a=name1, b=name2).ratio() >= border


def startProgress(title):
    global progress_x
    sys.stdout.write(title + ": \n| [" + "-" * 40 + "]\n|  ")
    sys.stdout.flush()
    progress_x = 0


def progress(x):
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x


def endProgress():
    sys.stdout.write("#" * (40 - progress_x) + "\n")
    sys.stdout.flush()
