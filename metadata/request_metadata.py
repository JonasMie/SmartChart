# coding=utf-8
import os
import re
import time

import discogs_client
import musicbrainzngs
import pandas as pd
import pyechonest.artist as echonest_artist
import pyechonest.config as echonest_config
import pyechonest.song as echonest_song
import pylast
import spotipy
from unidecode import unidecode

import config
import utils
from ArtistMetadata import ArtistMetadata
from TrackMetadata import TrackMetadata

track_md = None
artist_md = None

MUSICBRAINZ_LIMIT = 25
MUSICBRAINZ_TAG_BORDER = 0
DISCOGS_RELEASES_LIMIT = 10  # more would be better, but
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


def getMusicbrainzTrackMetadata(recording_list, exists_remix):
    for track in recording_list:
        if track_md is not None:
            track_md.musicbrainz_id = track['id']
        if 'length' in track:
            track_md.buffer['length'].append(int(track['length']))
        if 'tag-list' in track:
            for tag in track['tag-list']:
                if tag['count'] > MUSICBRAINZ_TAG_BORDER:
                    track_md.addTag(tag['name'])
    '''
    Avoid the property 'exists_remix' to be set to False (if at least one remix exists, this is true, if other service do not find any remix, then there still exists one
    '''
    if exists_remix is True:
        track_md.exists_remix = True


def getMusicbrainzArtistMetadata(artist):
    artist_md.musicbrainz_id = artist['id']
    if 'country' in artist:
        artist_md.country = artist['country']
    else:
        pass  # TODO: get country
    if 'area' in artist and 'iso-3166-1-code-list' in artist['area']:
        artist_md.area = artist['area']['iso-3166-1-code-list']
    else:
        pass  # TODO: get country
    if 'gender' in artist:
        artist_md.gender = artist['gender']
    if 'type' in artist:
        if artist['type'] == "Person":
            artist_type = 0
        elif artist['type'] == "Group":
            artist_type = 1
        else:
            artist_type = -1
        artist_md.addType(artist_type)

    if 'life-span' in artist:
        if 'end' in artist['life-span']:
            artist_md.life_span = utils.getActivity(start=artist['life-span']['begin'],
                                                    end=artist['life-span']['end'])
        else:
            artist_md.life_span = utils.getActivity(start=artist['life-span']['begin'])

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
        if 'text-representation' in release:
            artist_md.addLanguage(release['text-representation']['language'])
        if 'label-info-list' in release:
            for label in release['label-info-list']:
                if 'label' in label:
                    artist_md.addLabel({'name': label['label']['name'], 'release_count': None}, parent_label=None,
                                       sublabels=None)  # TODO: get missing data


def getMusicbrainzMetadata(track):
    global tries
    artist_id = None
    possible_tracks = []
    exists_remix = True
    offset = 0
    try:
        recordings = musicbrainzngs.search_recordings(
                "{} AND artist:\\{}".format(track[1].replace("(", "\(").replace(")", "\)"), track[0]),
                # TODO: escape chars
                limit=MUSICBRAINZ_LIMIT,
                offset=offset * MUSICBRAINZ_LIMIT)
        for recording in recordings["recording-list"]:
            for artist_credit in recording['artist-credit']:
                if 'artist' in artist_credit:
                    if artist_credit['artist']['name'] == track[0]:
                        if utils.checkTrackNamingConvention(recording['title'], track[1]):
                            possible_tracks.append(recording)
                            artist_id = artist_credit['artist']['id']
                    if utils.isTrackRemixByName(recording['title']):
                        exists_remix = True
        getMusicbrainzTrackMetadata(possible_tracks, exists_remix)
        if artist_id is None:
            choice = False
            artists = musicbrainzngs.search_artists(track[0],
                                                    ("artist", "begin", "end", "country", "ended", "gender",
                                                     "tag", "type", "area", "beginarea", "endarea"))
            for (i, artist) in enumerate(artists['artist-list']):
                '''
                resolve any disambiguations
                if the current artist has the same name as the next artist in the list, then let the user choose the right one
                '''
                if len(artists['artist-list']) - 1 > i and unidecode(artist['name']) == unidecode(
                        artists['artist-list'][i + 1]['name']) == track[0]:
                    choice = True
                    if i == 0:
                        print "Sorry, the artist '{0}' is ambigious, please chose the right one:\n[{1}] None of the options".format(
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
                            pass  # TODO: no artist fits
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
                elif unidecode(artist['name']) == track[0]:
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
        print "| The Musicbrainz service seems to be not available right now..."


def getDiscogsTrackMetadata(release):
    track_md.discogs_id = release.id

    for genre in release.genres:
        track_md.addGenre(genre)

    for style in release.data['styles']:
        track_md.addStyle(style)

    for track in release.data['tracklist']:
        if track['type_'] == 'track':
            '''
            Check if track has additional artists.
            If one of them represents a remixer, then the duration of the remix may manipulate
            the duration of the original songs, so only take the durations of the original song
            in account
            '''
        if 'extraartists' in track:
            for extraartist in track['extraartists']:
                if extraartist['role'] == 'Remix':  # TODO: check other roles
                    if not track_md.exists_remix:
                        track_md.exists_remix = True
                        break
                else:
                    if len(track['duration']) > 0:
                        track_md.buffer['length'].append(utils.getDuration(track['duration']))
        else:
            if len(track['duration']) > 0:
                track_md.buffer['length'].append(utils.getDuration(track['duration']))
    track_md.year = release.data['year']


def getDiscogsArtistMetadata(artist):
    release_count = 0
    artist_md.discogs_id = artist.id

    # try:
    artist_md.addType(1 if len(artist.members) > 1 else 0)
    artist_md.buffer['recording_count'].append(artist.releases.count)
    for group in artist.groups:  # TODO
        artist_md.addGroup(group)
    for release in artist.releases:
        for genre in release.genres:
            artist_md.addGenre(genre)
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
            # except


def getDiscogsMetadata(track):
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

                getDiscogsArtistMetadata(artist)
                for release in artist.releases:
                    if release.title == track[1]:  # TODO: check release types!!
                        getDiscogsTrackMetadata(release)
                        break
                break


def getEchonestTrackMetadata(tracks):
    for track in tracks:
        if track_md.echonest_id is None:
            track_md.echonest_id = track.id

        if hasattr(track, 'duration'):
            track_md.buffer['length'].append(track.duration)
        if hasattr(track, 'instrumentalness'):
            track_md.buffer['instrumentalness'].append(track.instrumentalness)
        if hasattr(track, 'speechiness'):
            track_md.buffer['speechiness'].append(track.speechiness)


def getEchonestArtistMetadata(artist):
    # artist_location?

    artist_md.echonest_id = artist.id
    artist_md.buffer['release_count'].append(artist.doc_counts['songs'])  # TODO: songs = release or recording or work?
    artist_md.buffer['news'].append(artist.doc_counts['news'])
    # more doc types: audio, biographies, blogs, images, reviews, video

    for term in artist.terms:
        track_md.addTag(term['name'], term['weight'])

    for years in artist.years_active:
        artist_md.addYearsActive(years)


def getEchonestMetadata(track):
    artist_id = None
    possible_artists = []
    possible_tracks = []

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
        getEchonestTrackMetadata(possible_tracks)
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
            getEchonestArtistMetadata(possible_artists[0])
    else:
        getEchonestArtistMetadata(echonest_artist.Artist(artist_id))


def getLastfmTrackMetadata(recording):
    track_md.lastfm_id = recording.get_id()
    for tag in recording.get_top_tags():
        track_md.addTag(tag.item.name, tag.weight)
    if recording.get_duration() > 0:
        track_md.buffer['length'].append(recording.get_duration())


def getLastfmArtistsMetadata(artist):
    # artist_md.lastfm_id = artist.get_id()   # Last.FM uses the musicbrainz-id
    artist_md.buffer['listener_count'].append(artist.get_listener_count())
    artist_md.buffer['play_count'].append(artist.get_playcount())
    for tag in artist.get_top_tags():
        artist_md.addTag(tag.item.name, tag.weight)


def getLastfmMetadata(track):
    recording = lastfm.get_track(track[0], track[1])
    try:
        getLastfmArtistsMetadata(recording.get_artist())
        getLastfmTrackMetadata(recording)
    except pylast.WSError:
        pass  # TODO


def getSpotifyTrackMetadata(possible_tracks):
    for track in possible_tracks:
        track_md.setSpotifyMarkets(track['available_markets'])
        track_md.buffer['length'].append(track['duration_ms'])


def getSpotifyArtistMetadata(artist):
    artist_md.spotify_id = artist['id']
    artist_md.buffer['followers'].append(artist['followers']['total'])
    for genre in artist['genres']:
        artist_md.addGenre(genre)
    artist_md.buffer['popularity'].append(artist['popularity'])
    # artist_md.addType(artist['type'])  # TODO: type==artist auch bei group?


def getSpotifyMetadata(track):
    artist_id = None
    possible_tracks = []
    recordings = spotify.search(q='artist:' + track[0] + ' track:' + track[1], type='track')
    for recording in recordings['tracks'][
        'items']:  # keys in recordings: items,next,href,limit,offset,total,previous
        for artist in recording['artists']:
            '''
            Spotify may return different versions of the same song, e.g. a Radio Edit, or the same song released on another label
            '''
            if artist['name'] == track[0] and (utils.checkTrackNamingConvention(recording['name'],
                                                                                track[1])):
                artist_id = artist['id']
                possible_tracks.append(recording)
    getSpotifyTrackMetadata(possible_tracks)
    if artist_id is None:
        artists = spotify.search(q='artist:' + track[0], type='artist')
        for artist in artists['artists']['items']:
            # TODO: multiple artists with same name => let user choose
            if artist['name'] == track[0]:
                getSpotifyArtistMetadata(artist)
                break
    else:
        getSpotifyArtistMetadata(spotify.artist(artist_id))


def getMetadata(fileList):
    global track_md, artist_md, tries
    tracks = []
    artists = []
    last_request = 0
    for artistTuple in fileList:
        artistName = artistTuple[0]
        for trackTuple in artistTuple[1]:
            track = [artistName, trackTuple[1]]
            print track
            print "---------- Colltecting data for {0} by {1} ----------\n|".format(track[1], artistName)
            track_md = TrackMetadata()
            artist_md = ArtistMetadata()

            print "| Collecting data from Musicbrainz..."
            if MAX_TRIES > tries or time.time() - last_request > TRY_AGAIN_AFTER:
                last_request = time.time()
                tries -= 1
                getMusicbrainzMetadata(track)
            else:
                print "|    The Musicbrainz-service was not reachable the last {} tries. Try again in {} seconds".format(
                        MAX_TRIES, int(TRY_AGAIN_AFTER - (time.time() - last_request)))
            print "| Collecting data from Discogs..."
            getDiscogsMetadata(track)
            print "| Collecting data from Echonest..."
            getEchonestMetadata(track)
            print "| Collecting data from Last.FM..."
            getLastfmMetadata(track)
            print "| Collecting data from Spotify..."
            getSpotifyMetadata(track)

            print "|"
            tracks.append(track_md.getData())
            artists.append(artist_md.getData())
    print "-------------------------------------------------------"

    artist_df = pd.DataFrame(artists)
    track_df = pd.DataFrame(tracks)

    artist_df.to_csv(os.path.join('features', 'meta_artist.csv'))
    track_df.to_csv(os.path.join('features', 'meta_track.csv'))
    return artist_df, track_df
