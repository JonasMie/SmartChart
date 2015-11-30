from MetadataBase import MetadataBase


class TrackMetadata(MetadataBase):
    def __init__(self):
        MetadataBase.__init__(self)
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

    def setSpotifyMarkets(self, markets):
        if 'DE' in markets:
            self.available_on_spotify_in_ger = True
        self.buffer['available_markets'].append(len(markets))

    def getData(self):
        return {
            'length': self.length,
            'instrumentalness:': self.instrumentalness,
            'speechiness': self.speechiness,
            'available_markets': self.available_markets,
            'year': self.year,
            'exists_remix': self.exists_remix,
            'available_on_spotify_in_ger': self.available_on_spotify_in_ger
        }

