import utils
from MetadataBase import MetadataBase


class ArtistMetadata(MetadataBase):
    def __init__(self):
        MetadataBase.__init__(self)
        self.musicbrainz_id = None
        self.discogs_id = None
        self.echonest_id = None
        self.lastfm_id = None
        self.spotify_id = None

        self.buffer = {
            'recording_count': [],
            'release_count': [],
            'work_count': [],
            'listener_count': [],
            'play_count': [],
            'followers': [],
            'popularity': [],
            'news': [],
            'years': []
        }
        self.country = None
        self.area = None
        self.gender = None

        '''
        artist_type ==
         0: person
         1: group
        -1: orchestra, choir, character and everything else
        '''
        self.artist_type = None
        self.life_span = None

        self.recording_count = None
        self.release_count = None
        self.work_count = None

        self.listener_count = None
        self.play_count = None
        self.followers = None
        self.popularity = None

        self.news = None
        self.groups = []  # todo

        self.years = None
        self.total_years = 0
        self.breaking_years = 0
        self.first_year = None
        self.last_year = None

        self.language = {'de': 0, 'en': 0, 'other': 0}

    def addLanguage(self, lang):
        if lang == 'eng':
            self.language['en'] += 1
        elif lang == 'deu':
            self.language['de'] += 1
        else:
            self.language['other'] += 1

    def addGroup(self, group):  # TODO
        pass

    def addType(self, artist_type):
        if self.artist_type is None:
            self.artist_type = artist_type
        elif self.artist_type is not artist_type:
            pass  # TODO: well, don't actually know what to do

    def addYearsActive(self, years):
        if self.first_year is None or years['start'] < self.first_year:
            self.first_year = years['start']
        if 'end' in years:
            self.breaking_years += years['end'] - years['start']
            if self.last_year is None or years['end'] > self.last_year:
                self.last_year = years['end']
        else:
            self.breaking_years += utils.getCurrentYear() - years['start']

    def normalize(self):
        if self.first_year is not None:
            if self.last_year is not None:
                self.total_years = self.last_year - self.first_year
            else:
                self.total_years = utils.getCurrentYear() - self.first_year
        MetadataBase.normalize(self)

    def getData(self):
        return {
            'country': self.country,
            'area': self.area,
            'gender': self.gender,
            'artist_type': self.artist_type,
            'life_span': self.life_span,
            'recording_count': self.recording_count,
            'release_count': self.release_count,
            'work_count': self.work_count,
            'listener_count': self.listener_count,
            'play_count': self.play_count,
            'followers': self.followers,
            'popularity': self.popularity,
            'news': self.news,
            'total_years': self.total_years,
            'breaking_years': self.breaking_years
        }