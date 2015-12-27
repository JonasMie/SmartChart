from MetadataBase import MetadataBase


class TrackMetadata(MetadataBase):
    def __init__(self, name, artistName):
        MetadataBase.__init__(self, name)
        self.artist_name = artistName
        self.musicbrainz_id = None
        self.discogs_id = None
        self.lastfm_id = None
        self.echonest_id = None
        self.spotify_id = None

        self.buffer = {
            'length': [],
            'instrumentalness': [],
            'speechiness': [],
            'available_markets': []
        }
        self.length = None

        self.role = None
        self.year = None
        self.instrumentalness = None
        self.speechiness = None
        self.exists_remix = False
        self.available_markets = None
        self.available_on_spotify_in_ger = False

        self.peakCategory = 7
        self.peakWeeks = 0

    def setSpotifyMarkets(self, markets):
        if 'DE' in markets:
            self.available_on_spotify_in_ger = True
        self.buffer['available_markets'].append(len(markets))

    def addChartData(self, chartData):
        self.peakCategory = chartData["target_peak_cat"]
        self.peakWeeks = chartData["target_peak_weeks"]
    def getData(self):
        return {
            'name': self.name,
            'clean_name': self.clean_name,
            'artist_name': self.artist_name,
            'length': self.length,
            'instrumentalness:': self.instrumentalness,
            'speechiness': self.speechiness,
            'available_markets': self.available_markets,
            'year': self.year,
            'exists_remix': self.exists_remix,
            'available_on_spotify_in_ger': self.available_on_spotify_in_ger,
            'genre_electronic': self.genre_electronic,
            'genre_pop': self.genre_pop,
            'genre_hiphop': self.genre_hiphop,
            'genre_rock': self.genre_rock,
            'genre_soul': self.genre_soul,
            'genre_jazz': self.genre_jazz,
            'genre_country': self.genre_country,
            'genre_other': self.genre_other,
            'musicbrainz_id': self.musicbrainz_id,
            'discogs_id': self.discogs_id,
            'lastfm_id': self.lastfm_id,
            'echonest_id': self.echonest_id,
            'spotify_id': self.spotify_id,
            'peakCategory': self.peakCategory,
            'peakWeeks': self.peakWeeks,
            'error': self.error
        }
