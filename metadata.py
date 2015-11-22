# coding=utf-8
import re

import pylast
import config
import musicbrainzngs
import discogs_client
import pyechonest.config as echonest_config
import pyechonest.artist as echonest_artist
import pyechonest.song as echonest_song
import spotipy
import utils
from unidecode import unidecode


# MUSICBRAINZ_LIMIT = None


def getMusicbrainzArtistMetadata(artist):
    artist_md = {}
    keys = ['country', 'area', 'gender', 'type', 'life-span']
    for key in keys:
        if key in artist:
            if key == 'area':
                if 'iso-3166-1-code-list' in artist[key]:
                    artist_md[key] = artist[key]['iso-3166-1-code-list']
                else:
                    pass  # TODO
            elif key == 'life-span':
                if 'end' in artist[key]:
                    artist_md[key] = utils.getActivity(start=artist[key]['begin'],
                                                       end=artist[key]['end'])
                else:
                    artist_md[key] = utils.getActivity(start=artist[key]['begin'])
            else:
                artist_md[key] = artist[key]
        else:
            artist_md[key] = None
    # TODO: tag-list
    return artist_md


def getMusicbrainzMetadata(tracklist):
    artist_md = None
    artist_id = None
    next = True
    offset = 0
    musicbrainzngs.set_useragent(config.name, config.version, config.contact)
    try:
        for track in tracklist:
            song_md = {'length': [], 'labels': set()}
            while next is True:
                recordings = musicbrainzngs.search_recordings(track[1])  # limit=MUSICBRAINZ_LIMIT,
                # offset=offset )#* MUSICBRAINZ_LIMIT)
                for recording in recordings["recording-list"]:
                    for artist_credit in recording['artist-credit']:
                        if 'artist' in artist_credit and artist_credit['artist']['name'] == track[0]:
                            next = False
                            offset += 1
                            song_md['length'].append(int(recording['length']))
                            artist_id = artist_credit['artist']['id']
                if next and len(recordings) < 25:  # MUSICBRAINZ_LIMIT:
                    next = False
            if artist_id is None:
                choice = False
                artists = musicbrainzngs.search_artists(track[0])
                for (i, artist) in enumerate(artists['artist-list']):
                    '''
                    resolve any disambiguations
                    if the current artist has the same name as the next artist in the list, then let the user choose the right one
                    '''
                    if len(artists['artist-list']) - 1 > i and unidecode(artist['name']) == unidecode(
                            artists['artist-list'][i + 1]['name']) == \
                            track[0]:
                        choice = True
                        if i == 0:
                            print "Sorry, the artist '{0}' is ambigious, please chose the right one:\n[{1}] None of the options".format(
                                artist['name'], i)
                        print u"[{0}] {1}: {2}".format(i + 1, artist['name'], artist['disambiguation'])
                    elif choice:
                        print u"[{0}] {1}: {2}".format(i + 1, artist['name'], artist['disambiguation'])
                        input = raw_input("Your choice: ")
                        try:
                            artist_int = int(input)
                            if artist_int == 0:
                                pass  # TODO: no artist fits
                            artist_md = getMusicbrainzArtistMetadata(artists['artist-list'][artist_int - 1])
                        except ValueError:
                            pass  # TODO
                        break
                    elif unidecode(artist['name']) == track[0]:
                        artist_md = getMusicbrainzArtistMetadata(artists)
                        break
            else:
                artist_md = getMusicbrainzArtistMetadata(musicbrainzngs.get_artist_by_id(artist_id)['artist'])
    except musicbrainzngs.NetworkError:
        pass  # TODO: handle error


def getDiscogsTrackMetadata(release):
    track_md = {}
    track_md['discogs_id'] = release.id
    track_md['genres'] = release.genres
    track_md['role'] = release.data['role']
    track_md['styles'] = release.data['styles']
    track_md['exists_remix'] = False

    duration = 0
    normal_tracks = 0
    for track in release.data['tracklist']:
        if track['type_']=='track':
            if 'extraartists' in track:
                for extraartist in track['extraartists']:
                    track_md['exists_remix'] = extraartist['role'] == 'Remix'  # TODO: check other roles
            else:
                normal_tracks += 1
                duration += 42 #     # TODO: parse track['duration'] to seconds

    if duration > 0:
        track_md['duration'] = duration/normal_tracks
    track_md['year'] = release.data['year']

    track_md['labels'] = []
    for labels in release.main_release.labels:
        track_md['labels'].append(labels.name)

    return track_md

def getDiscogsArtistMetadata(artist):
    artist_md = {}
    artist_md['discogs_id'] = artist.id
    artist_md['type'] = artist.data['type']
    artist_md['releases'] = artist.releases.count
    artist_md['groups'] = artist.groups
    return artist_md


def getDiscogsMetadata(tracklist):
    artist_id = None
    break_ = False
    discogs = discogs_client.Client('{0}/{1}'.format(config.name, config.version),
                                    user_token=config.api_keys['DISCOGS_KEY'])
    for track in tracklist:
        # recordings = discogs.search(track[1], type='release')
        # for recording in recordings:
        #     for artist_credit in recording.artists:
        #         if artist_credit.name == track[0]:
        #             artist_id = artist_credit.id
        #             print artist_id
        #             # TODO: process track data
        #             break_ = True
        #             break

        artists = discogs.search(track[0], type='artist')
        for artist in artists:
            '''
            problem: For artists with the same name discogs generates a suffix with the index, e.g. 'Adele', 'Adele (2)', 'Adele (3)',...
            and they don't provide the normal name of the artist in the artist-object.
            The solution is to filter out the suffix using regular expression in order to compare the name from the artist-object with the
            given track artist
            '''
            name = re.search("^(.*?)(\s\(\d\))?$", artist.name)
            if name:
                print name.group(1)
                if name.group(1) == track[0]:
                    artist_md = getDiscogsArtistMetadata(artist)
                    for release in artist.releases:
                        if release.title == track[1]:
                            track_md = getDiscogsTrackMetadata(release)
                            print artist_md
                            print track_md
                            break


def getEchonestMetadata(tracklist):
    artist_id = None
    echonest_config.ECHO_NEST_API_KEY = config.api_keys['ECHONEST_KEY']
    for track in tracklist:
        recordings = echonest_song.search(artist=track[0], title=track[1])
        for recording in recordings:
            if recording.title == track[1]:
                artist_id = recording.artist_id
                # TODO: process track data  (mean values => same song)
        if artist_id is None:
            pass  # TODO
        else:
            artist = echonest_artist.Artist(artist_id)


def getLastfmMetadata(tracklist):
    lastfm = pylast.LastFMNetwork(api_key=config.api_keys['LASTFM_KEY'])
    for track in tracklist:
        recordings = lastfm.get_track(track[0], track[1])
        try:
            print recordings.get_album()  # TODO: ?

        except pylast.WSError:
            # recordings = lastfm.search_for_track(track[0], track[1])
            pass


def getSpotifyMetadata(tracklist):
    artist_id = None
    spotify = spotipy.Spotify()
    for track in tracklist:
        recordings = spotify.search(q='track:' + track[1], type='track')
        for recording in recordings['tracks'][
            'items']:  # keys in recordings: items,next,href,limit,offset,total,previous
            for artist in recording['artists']:
                if artist['name'] == track[0]:
                    artist_id = artist['id']
                    pass  # TODO: process track data
        if artist_id is None:
            artist = spotify.search(q='artist:' + track[0], type='track')
            # TODO: process artist data
        else:
            artist = spotify.artist(artist_id)
            # TODO: process artist data


def getMetadata(tracklist):
    # getMusicbrainzMetadata(tracklist)
    getDiscogsMetadata(tracklist)
    # getEchonestMetadata(tracklist)
    # getLastfmMetadata(tracklist)
    # getSpotifyMetadata(tracklist)
