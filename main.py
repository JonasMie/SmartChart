# coding=utf-8
import requests
from bs4 import BeautifulSoup
import re
from metadata import getMetadata
from unidecode import unidecode


def getPeakPosition(artist, track):
    '''

    :param artist: string
    :param track: string
    :return: int
    '''

    '''
    Send a request with the track name as parameter to the search-URL  "https://www.offiziellecharts.de/suche"
    Parse the result and search for the URL of the first entry in the list
    '''
    search = requests.post("https://www.offiziellecharts.de/suche", data={"artist_search": track, "do_search": "do"})
    parsed_search = BeautifulSoup(search.text)
    try:
        track_url = parsed_search.find('table', class_='chart-table').findChildren()[8]['href']
    except AttributeError:
        return None

    '''
    Request the detailed track page and search for the track's peak position
    '''
    detail = requests.get("https://www.offiziellecharts.de%s" % track_url)
    parsed_detail = BeautifulSoup(detail.text, "html.parser")
    try:
        table_row = parsed_detail.find('table', class_='chart-table').findChildren()[6]
    except AttributeError:
        return None

    '''
    Check if the found table row contains the peak position
    '''
    if table_row.findChildren()[0].string == unicode("Höchstposition:", encoding='utf-8'):
        peak_re = re.search("[^\s]+", table_row.findChildren()[1].string)
        if peak_re:
            return peak_re.group()
    return None


if __name__ == "__main__":
    # getPeakPosition(None, "Cloud Rider")
    getMetadata([[unidecode("Paul Kalkbrenner"), "Cloud Rider"]])
    # getMetadata([[unidecode("Adele"), "Hello"]])
