# coding=utf-8

import requests
from bs4 import BeautifulSoup

from MIR.mir import *

CAT1 = 1
CAT2 = 2
CAT3 = 3
CAT4 = 4
CAT5 = 5
CAT6 = 6


def getPeakCategory(peak):
    if peak == 1:
        return CAT1
    elif peak < 6:
        return CAT2
    elif peak < 11:
        return CAT3
    elif peak < 51:
        return CAT4
    elif peak < 101:
        return CAT5
    else:
        return CAT6


def getPeakPosition(tracklist, Featurings=True):
    """

    :param tracklist:
    :param Featurings:
    :return: list
    """

    results = []
    for track in tracklist:
        '''
        Send a request with the track name as parameter to the search-URL  "https://www.offiziellecharts.de/suche"
        Parse the result and search for the URL of the first entry in the list
        '''

        track_results = {}
        dist_chart_peak = {CAT1: 0, CAT2: 0, CAT3: 0, CAT4: 0, CAT5: 0}
        total_chart_weeks = 0
        mean_chart_weeks = []
        mean_chart_peak = []
        target_peak_cat = CAT6
        target_url = None

        search = requests.post("https://www.offiziellecharts.de/suche",
                               data={"artist_search": track[0], "do_search": "do"})
        parsed_search = BeautifulSoup(search.text)
        '''
        Get the table with the search results
        We only have to continue if there are any results
        '''
        charts = parsed_search.find('table', class_='chart-table')
        if charts is not None:
            '''
            Get all table rows except the first one ("x Treffer in der Kategorie 'Single'")
            Then iterate through all rows
            '''
            charts = charts.find_all('tr', {'class': ''})
            for chart in charts:
                '''
                Check if the artist of the song matches the target artist
                '''
                if track[0] in chart.findChildren()[2].previousSibling.strip():
                    '''
                    Get the chart data of the song ("Wochen: X Peak: Y")
                    '''
                    chart_data = chart.findChildren()[6].text.split()
                    weeks = int(chart_data[1])
                    peak = int(chart_data[3])

                    '''
                    Get the relevant data:


                    dist_chart_peak:    the distribution of all the artist's chart songs corresponding to their target category
                    total_chart_weeks:  for how many weeks the artist has been in the charts with all of his songs in total
                    mean_chart_weeks:   mean chart position for all of the artist's songs in the charts
                    mean_chart_peak:    mean peak position for all of the artist's songs in the charts
                    '''
                    dist_chart_peak[getPeakCategory(peak)] += 1
                    total_chart_weeks += weeks
                    mean_chart_weeks.append(weeks)
                    mean_chart_peak.append(peak)

                    '''
                    if the current track equals the searched track, then get its peak category and save its detail url
                    (may be interesting for later use)
                    '''
                    a = chart.findChildren()[3]
                    if a.text == track[1]:
                        target_peak_cat = getPeakCategory(peak)
                        target_url = a['href']

        mean_chart_weeks = np.mean(mean_chart_weeks) if len(mean_chart_weeks) > 0 else 0
        mean_chart_peak = np.mean(mean_chart_peak) if len(mean_chart_peak) > 0 else 0
        track_results['artist_md'] = {'dist_chart_peak': dist_chart_peak, 'total_chart_weeks': total_chart_weeks,
                                      'mean_chart_weeks': mean_chart_weeks, 'mean_chart_peak': mean_chart_peak}
        track_results['target_peak_cat'] = target_peak_cat
        track_results['target_url'] = target_url
        results.append(track_results)

    return results

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


if __name__ == "__main__":
    #    track = raw_input("Which song to analyse?: ")
    #   artist = raw_input("Song is by which artist?: ")
    run()
