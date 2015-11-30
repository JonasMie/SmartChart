import numpy as np
import pandas as pd

DEFAULT_TAG_WEIGHT = .2


class MetadataBase:
    def __init__(self):
        self.labels = {'labels': {}, 'parent_labels': {}}
        self.tags = {}
        self.genres = {}
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

    def addGenre(self, genre_name):
        if genre_name in self.genres:
            self.genres[genre_name] += 1
        else:
            self.genres[genre_name] = 1

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