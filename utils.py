import datetime
import difflib
import re

from dateutil import parser

feat = re.compile(r"([\[(](?:ft?\.|featuring|feat(?:[\.]|))(.*)[\])])", re.I)


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
    return re.sub(r"([\[(](\s)*(?:ft?\.|featuring|feat(?:[\.]|))(.*)[\])])", '', track).strip().lower()


def is_similar(name1, name2, normalize=False, border=.9):
    if normalize:
        name1 = normalizeName(name1)
        name2 = normalizeName(name2)
    return difflib.SequenceMatcher(a=name1, b=name2).ratio() >= border
