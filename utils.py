import datetime


def getActivity(start, end=datetime.datetime.now(), format="%Y-%m-%d"):
    if isinstance(start, basestring):
        start = datetime.datetime.strptime(start, format)
    if isinstance(end, basestring):
        end = datetime.datetime.strptime(end, format)
    return (end-start).days