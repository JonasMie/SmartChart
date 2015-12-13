import datetime

from dateutil import parser


def getActivity(start, end=datetime.datetime.now(), format="%Y-%m-%d"):
    if isinstance(start, basestring):
        # start = datetime.datetime.strptime(start, format)
        start = parser.parse(start)
    if isinstance(end, basestring):
        end = parser.parse(end)
    return (end - start).days


def getCurrentYear():
    return datetime.datetime.now().year


def checkTrackNamingConvention(actual, target):
    variations = ["", " Radio Edit", " (Radio Edit)", " - Radio Edit", " -Radio Edit"]
    for variation in variations:
        if actual == target + variation:
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
