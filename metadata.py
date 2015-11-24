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
import numpy as np

MUSICBRAINZ_LIMIT = 25
LASTFM_TAG_WEIGHT_BORDER = 85
ECHONEST_TAG_WEIGHT_BORDER = .85


def getMusicbrainzTrackMetadata(recording_list):
    pass  # TODO


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
    results = []
    artist_md = None
    artist_id = None
    recording_list = []
    next = True
    offset = 0
    musicbrainzngs.set_useragent(config.name, config.version, config.contact)
    # try:
    for track in tracklist:
        song_md = {'length': [], 'labels': set()}
        recordings = musicbrainzngs.search_recordings(track[1] + " AND artist:\\" + track[0],
                                                      limit=MUSICBRAINZ_LIMIT,
                                                      offset=offset * MUSICBRAINZ_LIMIT)
        for recording in recordings["recording-list"]:
            for artist_credit in recording['artist-credit']:
                if 'artist' in artist_credit and artist_credit['artist']['name'] == track[0]:
                    recording_list.append(recording)
                    artist_id = artist_credit['artist']['id']
        track_md = getMusicbrainzTrackMetadata(recording_list)
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

        results.append({'artist_md': artist_md, 'track_md': track_md})

    return results


# except musicbrainzngs.NetworkError:
#     pass  # TODO: handle error


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
        if track['type_'] == 'track':
            if 'extraartists' in track:
                for extraartist in track['extraartists']:
                    track_md['exists_remix'] = extraartist['role'] == 'Remix'  # TODO: check other roles
            else:
                normal_tracks += 1
                duration += 42  # # TODO: parse track['duration'] to seconds

    if duration > 0:
        track_md['duration'] = duration / normal_tracks
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
    results = []
    track_md = None
    artist_md = None
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
                if name.group(1) == track[0]:

                    artist_md = getDiscogsArtistMetadata(artist)
                    for release in artist.releases:
                        if release.title == track[1]:
                            track_md = getDiscogsTrackMetadata(release)
                            print artist_md
                            print track_md
                            break
        results.append({'artist_md': artist_md, 'track_md': track_md})
    return results


def getEchonestTrackMetadata(tracks):
    track_md = {}
    keys = ['duration', 'instrumentalness', 'speechiness']
    for track in tracks:
        if 'echonest_id' not in track_md:
            track_md['echonest_id'] = track.id
        for key in keys:
            track_md[key] = (track.audio_summary[key] + track_md[key]) / 2 if key in track_md else track.audio_summary[
                key]  # TODO: mean-value wrong
    return track_md


def getEchonestArtistMetadata(artist):
    # artist_location?

    artist_md = {}
    artist_md['echonest_id'] = artist.id
    artist_md['songs'] = artist.doc_counts['songs']
    artist_md['news'] = artist.doc_counts['news']
    # more doc types: audio, biographies, blogs, images, reviews, video

    terms = []
    for term in artist.terms:
        if term['weight'] > ECHONEST_TAG_WEIGHT_BORDER:
            terms.append(term['name'])
    artist_md['terms'] = terms

    first_year = None
    last_year = None
    total_years = None
    breaking_years = 0

    for years in artist.years_active:
        if first_year is None or years['start'] < first_year:
            first_year = years['start']
        if 'end' in years:
            breaking_years += years['end'] - years['start']
            if last_year is None or years['end'] > last_year:
                last_year = years['end']
        else:
            breaking_years += utils.getCurrentYear() - years['start']
    if first_year is not None:
        if last_year is not None:
            total_years = last_year - first_year
        else:
            total_years = utils.getCurrentYear() - first_year

    artist_md['total_years'] = total_years
    artist_md['breaking_years'] = breaking_years if breaking_years > 0 else None
    return artist_md


def getEchonestMetadata(tracklist):
    results = []
    artist_md = None
    track_md = None
    artist_id = None
    possible_artists = []
    possible_tracks = []

    echonest_config.ECHO_NEST_API_KEY = config.api_keys['ECHONEST_KEY']

    for track in tracklist:
        '''
        Get all recordings from echonest with the corresponding artist and track-title
        '''
        recordings = echonest_song.search(artist=track[0], title=track[1])
        for recording in recordings:
            '''
            check if echonest found the correct title or if it returned just similar written tracks
            '''
            if recording.title == track[1] and recording.artist_name == track[0]:
                artist_id = recording.artist_id
                '''
                there may exist multiple entities for one song at echonest, so I collect all the matching songs
                '''
                possible_tracks.append(recording)
        '''
        if we found more than one track, get the mean metadata for all of them
        '''
        if len(possible_tracks) > 0:
            track_md = getEchonestTrackMetadata(possible_tracks)
        if artist_id is None:
            '''
            if echonest couldn't find a corresponding match for the given artist and track, there are chances it will still find
            only the artist by it's name
            '''
            artists = echonest_artist.search(name=track[0])
            for artist in artists:
                if artist.name == track[0]:
                    possible_artists.append(artist)
            if len(possible_artists) > 1:
                pass  # TODO let user choose
            elif len(possible_artists) == 1:
                artist_md = getEchonestArtistMetadata(possible_artists[0])
        else:
            artist_md = getEchonestArtistMetadata(echonest_artist.Artist(artist_id))
        results.append({'artist_md': artist_md, 'track_md': track_md})
    return results


def getLastfmTrackMetadata(recording):
    track_md = {}
    track_md['lastfm_id'] = recording.get_id()
    tags = []
    for tag in recording.get_top_tags():
        if int(tag.weight) > LASTFM_TAG_WEIGHT_BORDER:
            tags.append(tag.item.name)
    track_md['tags'] = tags
    track_md['duration'] = recording.get_duration() if recording.get_duration() > 0 else None
    return track_md


def getLastfmArtistsMetadata(artist):
    artist_md = {}
    artist_md['listener_count'] = artist.get_listener_count()
    artist_md['playcount'] = artist.get_playcount()
    tags = []
    for tag in artist.get_top_tags():
        if int(tag.weight) > LASTFM_TAG_WEIGHT_BORDER:
            tags.append(tag.item.name)
    artist_md['tags'] = tags
    return artist_md


def getLastfmMetadata(tracklist):
    results = []
    track_md = None
    artist_md = None
    lastfm = pylast.LastFMNetwork(api_key=config.api_keys['LASTFM_KEY'])
    for track in tracklist:
        recording = lastfm.get_track(track[0], track[1])
        try:
            artist_md = getLastfmArtistsMetadata(recording.get_artist())
            track_md = getLastfmTrackMetadata(recording)
        except pylast.WSError:
            pass  # TODO
        results.append({'artist_md': artist_md, 'track_md': track_md})
    return results


def getSpotifyTrackMetadata(possible_tracks):
    track_md = {}
    track_md['available_markets'] = []
    track_md['duration'] = []
    track_md['available_on_spotify_in_ger'] = False
    for track in possible_tracks:
        if 'DE' in track['available_markets']:
            track_md['available_on_spotify_in_ger'] = True
        track_md['available_markets'].append(len(track['available_markets']))
        track_md['duration'].append(track['duration_ms'])  # TODO: better in seconds?

    track_md['available_markets'] = np.mean(track_md['available_markets'])
    track_md['duration'] = np.mean(track_md['duration'])

    return track_md


def getSpotifyArtistMetadata(artist):
    artist_md = {}
    artist_md['spotify_id'] = artist['id']
    artist_md['spotify_followers'] = artist['followers']['total']
    artist_md['genres'] = artist['genres'] if len(artist['genres']) > 0 else None
    artist_md['popularity'] = artist['popularity']
    artist_md['type'] = artist['type']
    return artist_md


def getSpotifyMetadata(tracklist):
    results = []
    track_md = None
    artist_md = None
    artist_id = None
    possible_tracks = []
    spotify = spotipy.Spotify()
    for track in tracklist:
        recordings = spotify.search(q='artist:' + track[0] + ' track:' + track[1], type='track')
        for recording in recordings['tracks'][
            'items']:  # keys in recordings: items,next,href,limit,offset,total,previous
            for artist in recording['artists']:
                '''
                Spotify may return different versions of the same song, e.g. a Radio Edit, or the same song released on another label
                '''
                if artist['name'] == track[0] and (
                                recording['name'] == track[1] or utils.checkTrackNamingConvention(recording['name'],
                                                                                                  track[1])):
                    artist_id = artist['id']
                    possible_tracks.append(recording)
        track_md = getSpotifyTrackMetadata(possible_tracks)
        if artist_id is None:
            artists = spotify.search(q='artist:' + track[0], type='artist')
            for artist in artists['artists']['items']:
                # TODO: multiple artists with same name => let user choose
                if artist['name'] == track[0]:
                    artist_md = getSpotifyArtistMetadata(artist)
                    break
        else:
            artist_md = getSpotifyArtistMetadata(spotify.artist(artist_id))
        results.append({'artist_md': artist_md, 'track_md': track_md})
    return results


def getMetadata(tracklist):
    metadata = {}
    metadata['musicbrainz'] = getMusicbrainzMetadata(tracklist)
    metadata['discogs'] = getDiscogsMetadata(tracklist)
    metadata['echonest'] = getEchonestMetadata(tracklist)
    metadata['lastfm'] = getLastfmMetadata(tracklist)
    metadata['spotify'] = getSpotifyMetadata(tracklist)

    return metadata
