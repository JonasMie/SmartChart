# coding=utf-8
import re

import numpy as np

DEFAULT_TAG_WEIGHT = .5

all_genres = {
    "electronic": [
        "ambient", "breakbeat", "chiptune", "disco", "europop", "downtempo", "dnb", "drumnbass", "electro",
        "electronic", "electro swing", "electroacustic", "electronica", "edm", "electronicdancemusic", "electronicrock",
        "dancerock", "electropunk", "synthpop", "eurodance", "hardcore", "hardstyle", "house", "acidhouse", "deephouse",
        "electrohouse", "techhouse", "industrial", "techno", "detroittechno", "minimal", "trance", "garage", "ukgarage",
        "dubstep", "trap", "grime"
    ],
    "pop": [
        "pop", "chanson", "dancepop", "electropop", "europop", "jpop", "poprap", "poprock", "schlager", "worldbeat",
    ],
    "hiphop": [
        "hiphop", "eastcoasst", "gangstarap", "ghetto", "grime", "hippop", "rap", "black"
    ],
    "rock": [
        "rock", "neuedeutschehaerte", "ndh", "alternativerock", "grunge", "indierock", "industrial", "numetal",
        "garage",
        "heavymetal", "metal", "deathmetal", "poprock", "punk", "punkrock",
    ],
    "soul": [
        "rnb", "rb", "soul", "funk", "disco",
    ],
    "jazz": [
        "jazz", "boogiewoogie", "bossanova", "jazzblues", "swing"
    ],
    "country": [
        "country", "texas", "texascountry", "bluegrass", "nashwille", "western"
    ],
    # "Avant-garde": ["experimental", "experimentalmusic", "noise", "electroacoustic", "lo-fi"],
    # "Blues": ["blues"],
    # "caribbean": [
    #     "punta", "rasin", "reggae", "dancehall", "dub", "ragga", "reggaeton", "rumba", "ska", "2tone"
    # ],
    # "easy-listening": [
    #     "background", "elevator", "furniture", "lounge", "newage"
    # ]
}


class MetadataBase:
    def __init__(self, name):
        self.name = name
        self.clean_name = None
        self.labels = {'labels': {}, 'parent_labels': {}}
        self.tags = {}

        self.genre_electronic = None
        self.genre_pop = None
        self.genre_hiphop = None
        self.genre_rock = None
        self.genre_soul = None
        self.genre_jazz = None
        self.genre_country = None
        self.genre_other = None

        self.error = False

        self.styles = {}

    def addLabel(self, label, parent_label, sublabels):
        if label['name'] in self.labels['labels']:
            self.labels['labels'][label['name']]['count'] += 1
            if label['release_count'] is not None:
                self.labels['labels'][label['name']]['release_count'].append(label['release_count'])
            if sublabels is not None:
                self.labels['labels'][label['name']]['sublabels'].append(sublabels)
        else:
            release_count = [] if label['release_count'] is None else [label['release_count']]
            sublabels = [] if sublabels is None else [sublabels]

            self.labels['labels'][label['name']] = {'count': 1, 'release_count': release_count, 'sublabels': sublabels}

        if parent_label is not None:
            if parent_label['name'] in self.labels['parent_labels']:
                self.labels['parent_labels'][parent_label['name']]['count'] += 1
                if self.labels['parent_labels'][parent_label['name']]['release_count'] is not None:
                    self.labels['parent_labels'][parent_label['name']]['release_count'].append(label['release_count'])
            else:
                self.labels['parent_labels'][parent_label['name']] = {'count': 1,
                                                                      'release_count': [parent_label['release_count']]}

    def addGenres(self, genres):
        foundGenre = False
        for genre in genres:
            genre_ = re.sub('[^a-zA-Z0-9]', '', genre).lower()
            for subgenre in all_genres:
                if not foundGenre:
                    if genre_ in all_genres[subgenre]:
                        setattr(self, "genre_" + subgenre, True)
                        foundGenre = True
                    else:
                        setattr(self, "genre_" + subgenre, False)
                else:
                    setattr(self, "genre_" + subgenre, False)
            if foundGenre:
                setattr(self, "genre_other", False)
                return
        if not foundGenre:
            setattr(self, "genre_other", True)

    def addStyle(self, style_name):
        if style_name in self.styles:
            self.styles[style_name] += 1
        else:
            self.styles[style_name] = 1

    def addTag(self, tag_name, tag_weight=DEFAULT_TAG_WEIGHT):
        if tag_name in self.tags:
            self.tags[tag_name]['count'] += 1
            self.tags[tag_name]['weight'].append(tag_weight)
        else:
            self.tags[tag_name] = {'count': 1, 'weight': [tag_weight]}

    def normalize(self):
        for key, val in self.buffer.iteritems():
            if len(val) > 0:
                setattr(self, key, np.mean(val))
        return self
