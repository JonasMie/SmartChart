from __future__ import division

import sqlite3

import time
from termcolor import colored
from metadata.request_metadata import getMetadata
from MIR.mir import marsyas_analyse
import utils

conn = sqlite3.connect('data.db')


def collectData(fileList, tracks_found):
    c = conn.cursor()
    currTrack = 0
    for artist, tracks in fileList.iteritems():
        search_artist = True
        artist_id = None
        artistName = utils.normalizeName(artist)
        c.execute('SELECT * FROM artist WHERE clean_name=?', (artistName,))
        artist_ds = c.fetchone()

        if artist_ds is not None:
            artist_id = artist_ds[0]
            search_artist = False

        for track in tracks:
            if track[0] == "/Volumes/JONAS IPOD/iPod_Control/Music/F11/TDVZ.mp3":  # TODO: FIX
                continue
            currTrack += 1
            print colored(u"| => Collecting data for {0} by {1}".format(track[1], artistName), 'blue')
            print colored(u"| => Track {0} of {1}".format(currTrack, tracks_found),
                          'blue')
            print ("|")
            saveTrack = True
            if artist_id is not None:
                c.execute('SELECT * FROM track WHERE clean_name=? AND artist_id=?',
                          (utils.normalizeName(track[1]), artist_id))
                track_ = c.fetchone()
                if track_ is not None:
                    print u"|\n| Data is already existing    | Processed {0:.2f}%".format(
                            (currTrack / tracks_found) * 100)
                    print "|\n|-------------------------------------------------------"
                    continue
            track_md, artist_md = getMetadata(track, artistName, search_artist=search_artist)
            if search_artist and artist_md is not None:
                c.execute(
                        'INSERT INTO artist (name,clean_name,is_male,is_female,is_group,german,american,other_country,total_years,breaking_years,life_span,genre_electronic,genre_pop,genre_hiphop,genre_rock,genre_country,genre_jazz,genre_soul,genre_other,followers,listener,play_count,recordings,releases,works,popularity,news,mean_chart_peak, mean_chart_weeks,total_chart_weeks, musicbrainz_id, discogs_id, lastfm_id, echonest_id, spotify_id,error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (artist_md.name, artist_md.clean_name, artist_md.is_male, artist_md.is_female,
                         artist_md.is_group, artist_md.is_german, artist_md.is_american, artist_md.is_other_country,
                         artist_md.total_years, artist_md.breaking_years, artist_md.life_span,
                         artist_md.genre_electronic, artist_md.genre_pop, artist_md.genre_hiphop,
                         artist_md.genre_rock, artist_md.genre_country, artist_md.genre_jazz,
                         artist_md.genre_soul, artist_md.genre_other, artist_md.followers, artist_md.listener_count,
                         artist_md.play_count, artist_md.recording_count, artist_md.release_count, artist_md.work_count,
                         artist_md.popularity, artist_md.news, artist_md.meanChartPeak,
                         artist_md.meanChartWeeks, artist_md.totalChartWeeks,
                         artist_md.musicbrainz_id, artist_md.discogs_id, artist_md.lastfm_id, artist_md.echonest_id,
                         artist_md.spotify_id, artist_md.error))
                artist_id = c.lastrowid
                search_artist = False
            if saveTrack:
                track_mir = marsyas_analyse(track[0])
                c.execute(
                        'INSERT INTO track (name, clean_name, artist_id, musicbrainz_id, discogs_id, lastfm_id,echonest_id, spotify_id, genre_electronic, genre_pop, genre_hiphop, genre_rock, genre_country, genre_jazz, genre_soul, genre_other, year, is_2010s, is_2000s, is_1990s, is_1980s, is_other_decade, length, available_markets, available_on_spotify_in_ger, exists_remix, instrumentalness, speechiness, date, zcr, zcr_std, nrg, nrg_std,pow, pow_std,acr, acr_std,acr_lag, acr_lag_std,amdf,amdf_std, eoe, eoe_std, eoe_min,cent,cent_std,flx,flx_std,rlf,rlf_std, mfcc_0,mfcc_0_std,mfcc_1, mfcc_1_std,mfcc_2, mfcc_2_std, mfcc_3, mfcc_3_std,mfcc_4, mfcc_4_std,mfcc_5,mfcc_5_std,mfcc_6,mfcc_6_std,mfcc_7,mfcc_7_std,mfcc_8,mfcc_8_std,mfcc_9,mfcc_9_std,mfcc_10,mfcc_10_std,mfcc_11,mfcc_11_std,mfcc_12,mfcc_12_std, chr_0,chr_0_std,chr_1,chr_1_std,chr_2,chr_2_std,chr_3,chr_3_std,chr_4,chr_4_std,chr_5,chr_5_std,chr_6,chr_6_std,chr_7,chr_7_std,chr_8,chr_8_std,chr_9,chr_9_std,chr_10,chr_10_std,chr_11, chr_11_std,peak_cat, peak_weeks, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (track_md.name, track_md.clean_name, artist_id, track_md.musicbrainz_id, track_md.discogs_id,
                         track_md.lastfm_id, track_md.echonest_id, track_md.spotify_id, track_md.genre_electronic,
                         track_md.genre_pop, track_md.genre_hiphop, track_md.genre_rock, track_md.genre_country,
                         track_md.genre_jazz,
                         track_md.genre_soul, track_md.genre_other,
                         track_md.year,
                         track_md.is_2010s, track_md.is_2000s, track_md.is_1990s, track_md.is_1980s,
                         track_md.is_other_decade,
                         track_md.length,
                         track_md.available_markets,
                         track_md.available_on_spotify_in_ger, track_md.exists_remix, track_md.instrumentalness,
                         track_md.speechiness, time.time(),
                         track_mir['zcr'], track_mir['zcr_std'],
                         track_mir['nrg'], track_mir['nrg_std'],
                         track_mir['pow'], track_mir['pow_std'],
                         track_mir['acr'], track_mir['acr_std'],
                         track_mir['acr_lag'], track_mir['acr_lag_std'],
                         track_mir['amdf'], track_mir['amdf_std'],
                         track_mir['eoe'], track_mir['eoe_std'], track_mir['eoe_min'],
                         track_mir['cent'], track_mir['cent_std'],
                         track_mir['flx'], track_mir['flx_std'],
                         track_mir['rlf'], track_mir['rlf_std'],
                         track_mir['mfcc_0'], track_mir['mfcc_0_std'],
                         track_mir['mfcc_1'], track_mir['mfcc_1_std'],
                         track_mir['mfcc_2'], track_mir['mfcc_2_std'],
                         track_mir['mfcc_3'], track_mir['mfcc_3_std'],
                         track_mir['mfcc_4'], track_mir['mfcc_4_std'],
                         track_mir['mfcc_5'], track_mir['mfcc_5_std'],
                         track_mir['mfcc_6'], track_mir['mfcc_6_std'],
                         track_mir['mfcc_7'], track_mir['mfcc_7_std'],
                         track_mir['mfcc_8'], track_mir['mfcc_8_std'],
                         track_mir['mfcc_9'], track_mir['mfcc_9_std'],
                         track_mir['mfcc_10'], track_mir['mfcc_10_std'],
                         track_mir['mfcc_11'], track_mir['mfcc_11_std'],
                         track_mir['mfcc_12'], track_mir['mfcc_12_std'],
                         track_mir['chr_0'], track_mir['chr_0_std'],
                         track_mir['chr_1'], track_mir['chr_1_std'],
                         track_mir['chr_2'], track_mir['chr_2_std'],
                         track_mir['chr_3'], track_mir['chr_3_std'],
                         track_mir['chr_4'], track_mir['chr_4_std'],
                         track_mir['chr_5'], track_mir['chr_5_std'],
                         track_mir['chr_6'], track_mir['chr_6_std'],
                         track_mir['chr_7'], track_mir['chr_7_std'],
                         track_mir['chr_8'], track_mir['chr_8_std'],
                         track_mir['chr_9'], track_mir['chr_9_std'],
                         track_mir['chr_10'], track_mir['chr_10_std'],
                         track_mir['chr_11'], track_mir['chr_11_std'],
                         track_md.peakCategory, track_md.peakWeeks,
                         track_md.error))
            print "|\n|\n|"
            conn.commit()
            print colored(u"| Data saved with errors    | Processed {0:.2f}%".format((currTrack / tracks_found) * 100),
                          'yellow') if track_md.error else colored(
                    u"| Data saved successfully    | Processed {0:.2f}%".format((currTrack / tracks_found) * 100),
                    'green')
            print "|------------------------------------------------------"
    conn.close()
