import utils
from MetadataBase import MetadataBase


class ArtistMetadata(MetadataBase):
    def __init__(self, name):
        MetadataBase.__init__(self, name)

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
        self.is_german = None
        self.is_american = None
        self.is_other_country = None
        self.area = None

        self.is_male = None
        self.is_female = None
        self.is_group = None

        self.life_span = None

        self.recording_count = None
        self.release_count = None
        self.work_count = None

        self.listener_count = None
        self.play_count = None
        self.followers = None
        self.popularity = None

        self.news = None
        self.groups = []  # TODO

        self.years = None
        self.total_years = 0
        self.breaking_years = 0
        self.first_year = None
        self.last_year = None

        self.language = {'de': 0, 'en': 0, 'other': 0}

        self.distChartPeak = None
        self.totalChartWeeks = 0
        self.meanChartWeeks = 0
        self.meanChartPeak = 0

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
        return MetadataBase.normalize(self)

    def addChartData(self, chartData):
        self.distChartPeak = chartData['artist_md']['dist_chart_peak']
        self.totalChartWeeks = chartData['artist_md']['total_chart_weeks']
        self.meanChartWeeks = chartData['artist_md']['mean_chart_weeks']
        self.meanChartPeak = chartData['artist_md']['mean_chart_peak']

    def getData(self):
        return {
            'name': self.name,
            'clean_name': self.clean_name,
            'is_german': self.is_german,
            'is_american': self.is_american,
            'is_other_country': self.is_other_country,
            'area': self.area,
            'is_male': self.is_male,
            'is_female': self.is_female,
            'is_group': self.is_group,
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
            'breaking_years': self.breaking_years,
            'genre_electronic': self.genre_electronic,
            'genre_pop': self.genre_pop,
            'genre_hiphop': self.genre_hiphop,
            'genre_rock': self.genre_rock,
            'genre_soul': self.genre_soul,
            'genre_jazz': self.genre_jazz,
            'genre_country': self.genre_country,
            'genre_other': self.genre_other,
            # 'distChartPeak': self.distChartPeak,
            'totalChartWeeks': self.totalChartWeeks,
            'meanChartWeeks': self.meanChartWeeks,
            'meanChartPeak': self.meanChartPeak,
            'musicbrainz_id': self.musicbrainz_id,
            'discogs_id': self.discogs_id,
            'lastfm_id': self.lastfm_id,
            'echonest_id': self.echonest_id,
            'spotify_id': self.spotify_id,
            'error': self.error
        }
