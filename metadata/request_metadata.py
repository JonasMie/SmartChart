# coding=utf-8

import re
import socket
import time

import discogs_client
import musicbrainzngs
import numpy as np
import pyechonest.artist as echonest_artist
import pyechonest.config as echonest_config
import pyechonest.song as echonest_song
import pylast
import requests
import spotipy
from bs4 import BeautifulSoup
from termcolor import colored

import config
import utils
from ArtistMetadata import ArtistMetadata
from TrackMetadata import TrackMetadata

track_md = None
artist_md = None

MUSICBRAINZ_LIMIT = 25
MUSICBRAINZ_TAG_BORDER = 0
DISCOGS_RELEASES_LIMIT = 20
LASTFM_TAG_WEIGHT_BORDER = 85
ECHONEST_TAG_WEIGHT_BORDER = .85
MAX_TRIES = 1
TRY_AGAIN_AFTER = 300
tries = 0

musicbrainzngs.set_useragent(config.name, config.version, config.contact)
discogs = discogs_client.Client('{0}/{1}'.format(config.name, config.version),
                                user_token=config.api_keys['DISCOGS_KEY'])
lastfm = pylast.LastFMNetwork(api_key=config.api_keys['LASTFM_KEY'])
echonest_config.ECHO_NEST_API_KEY = config.api_keys['ECHONEST_KEY']
spotify = spotipy.Spotify()

CAT1 = 1
CAT2 = 2
CAT3 = 3
CAT4 = 4
CAT5 = 5
CAT6 = 6
CAT0 = 0


def getMusicbrainzTrackMetadata(recording_list, exists_remix):
    for track in recording_list:
        if track_md.musicbrainz_id is None:
            track_md.musicbrainz_id = track['id']
        if 'length' in track:
            track_md.buffer['length'].append(int(track['length']) / 1000)
        if 'tag-list' in track:
            for tag in track['tag-list']:
                if tag['count'] > MUSICBRAINZ_TAG_BORDER:
                    track_md.addTag(tag['name'])
    '''
    Avoid the property 'exists_remix' to be set to False (if at least one remix exists, this is true, if other service do not find any remix, then there still exists one
    '''
    if exists_remix:
        track_md.exists_remix = True


def getMusicbrainzArtistMetadata(artist):
    artist_md.musicbrainz_id = artist['id']
    if 'country' in artist:
        if artist['country'] == 'US':
            artist_md.is_american = True
            artist_md.is_german = False
            artist_md.is_other_country = False
        elif artist['country'] == 'DE':
            artist_md.is_german = True
            artist_md.is_american = False
            artist_md.is_other_country = False
        else:
            artist_md.is_other_country = True
            artist_md.is_american = False
            artist_md.is_german = False
        artist_md.country = artist['country']
    else:
        pass  # TODO: get country
    if 'area' in artist and 'iso-3166-1-code-list' in artist['area']:
        artist_md.area = artist['area']['iso-3166-1-code-list']
    else:
        pass  # TODO: get country
    if 'type' in artist:
        if artist['type'] != "Person":
            artist_md.is_group = True
            artist_md.is_male = False
            artist_md.is_female = False
        else:
            artist_md.is_group = False
    if 'gender' in artist:
        if artist['gender'] == 'Male':
            artist_md.is_male = True
            artist_md.is_female = False
            artist_md.is_group = False
        elif artist['gender'] == 'Female':
            artist_md.is_male = False
            artist_md.is_female = True
            artist_md.is_group = False

    if 'life-span' in artist:
        if 'begin' in artist['life-span']:
            if 'end' in artist['life-span']:
                artist_md.life_span = utils.getActivity(start=artist['life-span']['begin'],
                                                        end=artist['life-span']['end'])
            else:
                try:
                    artist_md.life_span = utils.getActivity(start=artist['life-span']['begin'])
                except ValueError:
                    print colored("| Date error...", 'red')
                    artist_md.error = True

    '''
    musicbrainz uses a rating based on 5 (0 = bad, 5= good) but I want a float between 0 and 100
    '''
    if 'rating' in artist:
        artist_md.buffer['popularity'].append(float(artist['rating']['rating']) * 20)
    artist_md.buffer['recording_count'].append(artist['recording-count'])
    artist_md.buffer['release_count'].append(artist['release-count'])
    artist_md.buffer['work_count'].append(artist['work-count'])
    if 'tag-list' in artist:
        for tag in artist['tag-list']:
            artist_md.addTag(tag['name'])  # TODO: 'count' instead of weight

    releases = musicbrainzngs.browse_releases(artist['id'], includes=['labels'], limit=50)
    for release in releases['release-list']:
        if 'text-representation' in release and 'language' in release['text-representation']:
            artist_md.addLanguage(release['text-representation']['language'])
        if 'label-info-list' in release:
            for label in release['label-info-list']:
                if 'label' in label:
                    artist_md.addLabel({'name': label['label']['name'], 'release_count': None}, parent_label=None,
                                       sublabels=None)  # TODO: get missing data


def getMusicbrainzMetadata(track, search_artist=True):
    global tries
    artist_id = None
    possible_tracks = []
    exists_remix = True
    offset = 0
    try:
        recordings = musicbrainzngs.search_recordings(query=track[1].replace('/', ''),
                                                      artistname=track[0].replace('/', ''),
                                                      limit=MUSICBRAINZ_LIMIT,
                                                      offset=offset * MUSICBRAINZ_LIMIT)
        for recording in recordings["recording-list"]:
            for artist_credit in recording['artist-credit']:
                if 'artist' in artist_credit and utils.is_similar(artist_credit['artist']['name'].lower(), track[0],
                                                                  normalize=True):
                    if utils.checkTrackNamingConvention(recording['title'].lower(), track[1]):
                        if 'release-list' in recording:
                            for release in recording['release-list']:
                                if 'status' in release:
                                    if release['status'] == 'Official':
                                        possible_tracks.append(recording)
                                        if artist_id is None:
                                            artist_id = artist_credit['artist']['id']
                                    elif release['status'] == 'Bootleg':
                                        exists_remix = True
                                else:
                                    if utils.isTrackRemixByName(recording['title']):
                                        exists_remix = True
                                    else:
                                        possible_tracks.append(recording)
        getMusicbrainzTrackMetadata(possible_tracks, exists_remix)
        if search_artist:
            if artist_id is None:
                choice = False
                artists = musicbrainzngs.search_artists(track[0].replace('/', ''),
                                                        ("artist", "begin", "end", "country", "ended", "gender",
                                                         "tag", "type", "area", "beginarea", "endarea"))
                for (i, artist) in enumerate(artists['artist-list']):
                    '''
                    resolve any disambiguations
                    if the current artist has the same name as the next artist in the list, then let the user choose the right one
                    '''
                    if len(artists['artist-list']) - 1 > i and utils.is_similar(artist['name'],
                                                                                artists['artist-list'][i + 1]['name'],
                                                                                normalize=True) and utils.is_similar(
                            track[0], artist['name'], normalize=True):
                        choice = True
                        if i == 0:
                            print u"Sorry, the artist '{0}' is ambigious, please chose the right one:\n[{1}] None of the options".format(
                                    artist['name'], i)
                        print u"[{0}] {1}: {2}".format(i + 1, artist['name'],
                                                       artist[
                                                           'disambiguation'] if 'disambiguation' in artist else "no description")
                    elif choice:
                        print u"[{0}] {1}: {2}".format(i + 1, artist['name'], artist[
                            'disambiguation'] if 'disambiguation' in artist else "no description")
                        input = raw_input("Your choice: ")
                        try:
                            artist_int = int(input)
                            if artist_int == 0:
                                return
                            # FIXME: why does musicbrainzngs.search_artist() not provide this information? => double request necessary
                            # getMusicbrainzArtistMetadata(artists['artist-list'][artist_int - 1])
                            getMusicbrainzArtistMetadata(
                                    musicbrainzngs.get_artist_by_id(artists['artist-list'][artist_int - 1]['id'],
                                                                    ['recordings', 'releases',
                                                                     'release-groups', 'works',
                                                                     'aliases', 'artist-rels',
                                                                     'label-rels', 'tags', 'ratings'])[
                                        'artist'])
                        except ValueError:
                            pass  # TODO
                        break
                    elif utils.is_similar(artist['name'], track[0], normalize=True):
                        # FIXME: why does musicbrainzngs.search_artist() not provide this information? => double request necessary
                        getMusicbrainzArtistMetadata(musicbrainzngs.get_artist_by_id(artist['id'],
                                                                                     ['recordings', 'releases',
                                                                                      'release-groups', 'works',
                                                                                      'aliases', 'artist-rels',
                                                                                      'label-rels', 'tags', 'ratings'])[
                                                         'artist'])
                        break
            else:
                getMusicbrainzArtistMetadata(musicbrainzngs.get_artist_by_id(artist_id,
                                                                             ['recordings', 'releases',
                                                                              'release-groups', 'works',
                                                                              'aliases', 'artist-rels',
                                                                              'label-rels', 'tags', 'ratings'])[
                                                 'artist'])
    except musicbrainzngs.NetworkError:
        tries += 1
        track_md.error = True
        print colored("| The Musicbrainz service seems to be not available right now...", 'red')
    except musicbrainzngs.musicbrainz.ResponseError:  # Todo
        track_md.error = True
        print colored("| The Musicbrainz service sent a flawed response back...", 'red')


def getDiscogsTrackMetadata(release, track):
    track_md.discogs_id = release.id
    track_md.addGenres(release.data['genres'])

    if 'styles' in release.data:
        for style in release.data['styles']:
            track_md.addStyle(style)

    if len(track['duration']) > 0:
        track_md.buffer['length'].append(utils.getDuration(track['duration']))

    for label in release.labels:
        parent_label = {'name': label.parent_label.name,
                        'release_count': label.parent_label.releases.count} if label.parent_label is not None else None
        track_md.addLabel({'name': label.name, 'release_count': label.releases.count},
                          parent_label=parent_label,
                          sublabels=len(label.sublabels))
    track_md.addYear(release.data['year'])


def getDiscogsArtistMetadata(artist):
    release_count = 0
    artist_md.discogs_id = artist.id
    if len(artist.members) > 1:
        artist_md.is_group = True
        artist_md.is_male = False
        artist_md.is_female = False
    else:
        artist_md.is_group = False
    # artist_md.buffer['release_count'].append(artist.releases.count)  # TODO: check recording/release/work
    for release in artist.releases:
        if artist_md.genre_other is None or artist_md.genre_other:
            artist_md.addGenres(release.genres)
        if 'labels' in release.data:
            for label in release.labels:
                parent_label = {'name': label.parent_label.name,
                                'release_count': label.parent_label.releases.count} if label.parent_label is not None else None
                artist_md.addLabel({'name': label.name, 'release_count': label.releases.count},
                                   parent_label=parent_label,
                                   sublabels=len(label.sublabels))
        if 'year' in release.data and release.data['year'] > 0:
            artist_md.buffer['years'].append(release.data['year'])
        release_count += 1
        if release_count >= DISCOGS_RELEASES_LIMIT:
            break


def getDiscogsMetadata(track, search_artist=True):
    # artists = discogs.search(track[1], type='release', artist=track[0])

    # TODO: discogs offers a really really REALLY weird search functionality. You never get what you expect to get (if you get anything at all...)
    # The best, most performant and according to discogs 'correct' way to query the data is the call above this comment, but... well, it does not work, so I try to get the best results with
    # this call:
    try:
        releases = discogs.search(u"{0}+{1}".format(track[0], track[1]), type="release")
        processed_artist = False
        checked_releases = 0
        for release in releases:
            '''
            problem: For artists with the same name discogs generates a suffix with the index, e.g. '# TODO: discogs offers a really really REALLY weird search functionality. You never get what you expect to get (if you get anything at all...) Adele', 'Adele (2)', 'Adele (3)',...
            and they don't provide the normal name of the artist in the artist-object.
            The solution is to filter out the suffix using regular expression in order to compare the name from the artist-object with the
            given track artist
            '''
            checked_releases += 1
            if checked_releases > DISCOGS_RELEASES_LIMIT:
                break
            for artist in release.artists:
                name = re.search("^(.*?)(\s\(\d\))?$", artist.name)
                if name:
                    if utils.is_similar(name.group(1).lower(), track[0]):
                        if search_artist and not processed_artist:
                            getDiscogsArtistMetadata(artist)
                            processed_artist = True
                        for track_obj in release.data['tracklist']:
                            if utils.is_similar(track_obj['title'], track[1], normalize=True):
                                getDiscogsTrackMetadata(release, track_obj)
                                return

        if not processed_artist and search_artist:
            artists = discogs.search(track[0], type="artist")
            for artist in artists:
                if utils.is_similar(artist.name.lower(), track[0]):
                    getDiscogsArtistMetadata(artist)
                    break
    except requests.exceptions.SSLError:
        track_md.error = True
        print colored("| SSL error...", 'red')
    except discogs_client.exceptions.HTTPError:
        track_md.error = True
        print colored("| Internal discogs-server error...", 'red')


def getEchonestTrackMetadata(tracks):
    for track in tracks:
        if track_md.echonest_id is None:
            track_md.echonest_id = track.id

        if track.audio_summary['duration'] is not None:
            track_md.buffer['length'].append(track.audio_summary['duration'])
        if track.audio_summary['instrumentalness'] is not None:
            track_md.buffer['instrumentalness'].append(track.audio_summary['instrumentalness'])
        if track.audio_summary['speechiness'] is not None:
            track_md.buffer['speechiness'].append(track.audio_summary['speechiness'])
            # TODO track['song_type']


def getEchonestArtistMetadata(artist):
    artist_md.echonest_id = artist.id
    # artist_md.buffer['recording_count'].append(artist.doc_counts['songs'])
    artist_md.buffer['news'].append(artist.doc_counts['news'])

    for term in artist.terms:
        track_md.addTag(term['name'], term['weight'])

    for years in artist.years_active:
        artist_md.addYearsActive(years)


def getEchonestMetadata(track, search_artist=True):
    artist_id = None
    possible_artists = []
    possible_tracks = []

    try:
        '''
        Get all recordings from echonest with the corresponding artist and track-title
        '''
        recordings = echonest_song.search(artist=track[0], title=track[1])
        for recording in recordings:
            '''
            check if echonest found the correct title or if it returned just similar written tracks
            '''
            if utils.is_similar(recording.title, track[1], normalize=True) and utils.is_similar(
                    recording.artist_name.lower(),
                    track[0]):
                if artist_id is None:
                    artist_id = recording.artist_id
                '''
                there may exist multiple entities for one song at echonest, so I collect all the matching songs
                '''
                possible_tracks.append(recording)
        '''
        if we found more than one track, get the mean metadata for all of them
        '''
        if len(possible_tracks) > 0:
            getEchonestTrackMetadata(possible_tracks)

        if search_artist:
            if artist_id is None:
                '''
                if echonest couldn't find a corresponding match for the given artist and track, there are chances it will still find
                only the artist by it's name
                '''
                artists = echonest_artist.search(name=track[0])
                for artist in artists:
                    if utils.is_similar(artist.name, track[0], normalize=True):
                        possible_artists.append(artist)
                if len(possible_artists) > 1:
                    pass  # TODO let user choose
                elif len(possible_artists) == 1:
                    getEchonestArtistMetadata(possible_artists[0])
            else:
                getEchonestArtistMetadata(echonest_artist.Artist(artist_id))
    except socket.timeout:
        track_md.error = True
        print colored("| Socket timeout...", 'red')


def getLastfmTrackMetadata(recording):
    try:
        track_md.lastfm_id = 1  # recording.get_id()
        for tag in recording.get_top_tags():
            track_md.addTag(tag.item.name, tag.weight)
        if recording.get_duration() > 0:
            track_md.buffer['length'].append(recording.get_duration() / 1000)
    except pylast.MalformedResponseError:
        track_md.error = True
        print colored("| LastFM error...", 'red')


def getLastfmArtistsMetadata(artist):
    # artist_md.lastfm_id = artist.get_id()   # Last.FM uses the musicbrainz-id
    artist_md.buffer['listener_count'].append(artist.get_listener_count())
    artist_md.buffer['play_count'].append(artist.get_playcount())
    for tag in artist.get_top_tags():
        artist_md.addTag(tag.item.name, tag.weight)


def getLastfmMetadata(track, search_artist=True):
    try:
        recording = lastfm.get_track(track[0], track[1])
        if search_artist:
            getLastfmArtistsMetadata(recording.get_artist())
        getLastfmTrackMetadata(recording)
    except pylast.WSError, pylast.MalformedResponseError:
        track_md.error = True
        print colored("| LastFM error...", 'red')


def getSpotifyTrackMetadata(possible_tracks):
    for track in possible_tracks:
        track_md.spotify_id = 1
        track_md.setSpotifyMarkets(track['available_markets'])
        track_md.buffer['length'].append(track['duration_ms'] / 1000)


def getSpotifyArtistMetadata(artist):
    artist_md.spotify_id = artist['id']
    if artist['followers']['total'] is not None:
        artist_md.buffer['followers'].append(artist['followers']['total'])
    if artist_md.genre_other is None or artist_md.genre_other:
        artist_md.addGenres(artist['genres'])

    artist_md.buffer['popularity'].append(artist['popularity'])
    # artist_md.addType(artist['type'])  # TODO: type==artist auch bei group?


def getSpotifyMetadata(track, search_artist=True):
    artist_id = None
    possible_tracks = []
    recordings = spotify.search(q='artist:' + track[0] + ' track:' + track[1], type='track')
    for recording in recordings['tracks']['items']:  # keys in recordings: items,next,href,limit,offset,total,previous
        for artist in recording['artists']:
            '''
            Spotify may return different versions of the same song, e.g. a Radio Edit, or the same song released on another label
            '''
            if utils.is_similar(artist['name'].lower(), track[0]) and (
                    utils.checkTrackNamingConvention(recording['name'].lower(),
                                                     track[1], normalize=True)):
                if artist_id is None:
                    artist_id = artist['id']
                possible_tracks.append(recording)
    getSpotifyTrackMetadata(possible_tracks)
    if search_artist:
        if artist_id is None:
            artists = spotify.search(q='artist:' + track[0], type='artist')
            for artist in artists['artists']['items']:
                # TODO: multiple artists with same name => let user choose
                if utils.is_similar(artist['name'].lower(), track[0]):
                    getSpotifyArtistMetadata(artist)
                    break
        else:
            getSpotifyArtistMetadata(spotify.artist(artist_id))


def getPeakCategory(peak):
    if peak == 1:
        return CAT1
    elif peak < 6:
        return CAT2
    elif peak < 11:
        return CAT3
    elif peak < 21:
        return CAT4
    elif peak < 51:
        return CAT5
    elif peak < 101:
        return CAT6
    else:
        return CAT0


def getPeakPosition(tracklist, searchArtist=False, Featurings=True):
    """

    :param tracklist:
    :param Featurings:
    :return: list
    """

    print "| Searching for peak position..."
    results = []
    for track in tracklist:
        '''
        Send a request with the track name as parameter to the search-URL  "https://www.offiziellecharts.de/suche"
        Parse the result and search for the URL of the first entry in the list
        '''

        track_results = {}
        dist_chart_peak = {CAT1: 0, CAT2: 0, CAT3: 0, CAT4: 0, CAT5: 0, CAT6: 0}
        total_chart_weeks = 0
        total_albums_chart_weeks = 0
        mean_chart_weeks = []
        mean_albums_chart_weeks = []
        mean_chart_peak = []
        mean_albums_chart_peak = []
        target_peak_cat = CAT0
        target_peak_weeks = 0
        target_url = None

        search = requests.post("https://www.offiziellecharts.de/suche",
                               data={"artist_search": track[0], "do_search": "do"})
        parsed_search = BeautifulSoup(search.text, 'html.parser')
        '''
        Get the table with the search results
        We only have to continue if there are any results
        '''
        singles_table = parsed_search.find('div', {"id": 'searchtab-0', "class": 'active'})
        if singles_table is not None:
            charts = singles_table.find('table', class_='chart-table')
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
                    if track[0] in chart.findChildren()[
                        2].previousSibling.strip().lower():  # TODO: offiziellecharts.de filters out all featuring data
                        '''
                        Get the chart data of the song ("Wochen: X Peak: Y")
                        '''
                        chart_data = chart.findChildren()[6].text.split()
                        weeks = int(chart_data[1])
                        peak = int(chart_data[3])
                        if searchArtist:
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
                        if utils.is_similar(a.text, track[1], normalize=True):
                            target_peak_cat = getPeakCategory(peak)
                            target_peak_weeks = weeks
                            target_url = a['href']
                            if not searchArtist:
                                break
        if searchArtist:
            albums_table = parsed_search.find('div', {"id": 'searchtab-1'})
            if albums_table is None:
                albums_table = parsed_search.find('div', {"id": 'searchtab-0', "class": 'tab-pane'})
            if albums_table is not None:
                albums_charts = albums_table.find('table', class_='chart-table')
                if albums_charts is not None:
                    '''
                    Get all table rows except the first one ("x Treffer in der Kategorie 'Single'")
                    Then iterate through all rows
                    '''
                    albums_charts = albums_charts.find_all('tr', {'class': ''})
                    for album_chart in albums_charts:
                        '''
                        Check if the artist of the song matches the target artist
                        '''
                        if track[0] in album_chart.findChildren()[
                            2].previousSibling.strip().lower():  # TODO: offiziellecharts.de filters out all featuring data
                            '''
                            Get the chart data of the song ("Wochen: X Peak: Y")
                            '''
                            album_chart_data = album_chart.findChildren()[6].text.split()
                            weeks = int(album_chart_data[1])
                            peak = int(album_chart_data[3])

                            # dist_chart_peak[getPeakCategory(peak)] += 1
                            total_albums_chart_weeks += weeks
                            mean_albums_chart_weeks.append(weeks)
                            mean_albums_chart_peak.append(peak)

        if searchArtist:
            mean_chart_weeks = np.mean(mean_chart_weeks) if len(mean_chart_weeks) > 0 else 0
            mean_chart_peak = getPeakCategory(np.mean(mean_chart_peak)) if len(mean_chart_peak) > 0 else CAT0

            mean_album_chart_weeks = np.mean(mean_albums_chart_weeks) if len(mean_albums_chart_weeks) > 0 else 0
            mean_album_chart_peak = getPeakCategory(np.mean(mean_albums_chart_peak)) if len(
                    mean_albums_chart_peak) > 0 else CAT0
            track_results['artist_md'] = {'dist_chart_peak': dist_chart_peak, 'total_chart_weeks': total_chart_weeks,
                                          'mean_chart_weeks': mean_chart_weeks, 'mean_chart_peak': mean_chart_peak,
                                          'total_album_chart_weeks': total_albums_chart_weeks,
                                          'mean_album_chart_weeks': mean_album_chart_weeks,
                                          'mean_album_chart_peak': mean_album_chart_peak}
        track_results['target_peak_cat'] = target_peak_cat
        track_results['target_peak_weeks'] = target_peak_weeks
        track_results['target_url'] = target_url
        results.append(track_results)

    return results


def getMetadata(trackName, artistName, search_artist):
    global track_md, artist_md, tries
    tracks = []
    artists = []
    last_request = 0
    track_md = TrackMetadata(trackName, artistName)
    if search_artist:
        artist_md = ArtistMetadata(artistName)
        artist_md.clean_name = utils.normalizeName(artistName)

    track_md.clean_name = utils.normalizeName(trackName)

    track = [artistName.lower(), track_md.clean_name]

    print "| Collecting data from Musicbrainz..."
    if MAX_TRIES > tries or time.time() - last_request > TRY_AGAIN_AFTER:
        last_request = time.time()
        # tries -= 1
        getMusicbrainzMetadata(track, search_artist)
    else:
        print "|    The Musicbrainz-service was not reachable the last {} tries. Try again in {} seconds".format(
                MAX_TRIES, int(TRY_AGAIN_AFTER - (time.time() - last_request)))
    print "| Collecting data from Discogs..."
    getDiscogsMetadata(track, search_artist)
    print "| Collecting data from Echonest..."
    getEchonestMetadata(track, search_artist)
    print "| Collecting data from Last.FM..."
    getLastfmMetadata(track, search_artist)
    print "| Collecting data from Spotify..."
    getSpotifyMetadata(track, search_artist)

    print "|"

    chartData = getPeakPosition([[track[0], track[1]]], searchArtist=search_artist)[0]
    track_md.addChartData(chartData)
    if search_artist:
        artist_md.addChartData(chartData)

    tracks.append(track_md.normalize().getData())
    if search_artist:
        artists.append(artist_md.normalize().getData())

    print "|"
    return track_md, artist_md
