import datetime


def getActivity(start, end=datetime.datetime.now(), format="%Y-%m-%d"):
    if isinstance(start, basestring):
        start = datetime.datetime.strptime(start, format)
    if isinstance(end, basestring):
        end = datetime.datetime.strptime(end, format)
    return (end - start).days


def getCurrentYear():
    return datetime.datetime.now().year


def checkTrackNamingConvention(actual, target):
    variations = [" Radio Edit", " (Radio Edit)", " - Radio Edit", " -Radio Edit"]
    for variation in variations:
        if actual == target + variation:
            return True
    return False

def isTrackRemixByName(track):
    return "Remix" in track