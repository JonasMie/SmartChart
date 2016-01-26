import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.preprocessing import Imputer

con = sqlite3.connect('data.db')

colors = ['r', 'b', 'g', 'y']
ylabels = {'mse': 'mean squared error', 'mcc': 'mean categorical cross-entropy'}


def getDecisionData(n_rows, ratio=1):
    non_charts = pd.read_sql_query(
            "SELECT track.genre_electronic, track.genre_pop, track.genre_hiphop, track.genre_rock, track.genre_other, track.is_1980s, track.is_1990s, track.is_2000s, track.is_2010s, track.is_other_decade,  track.length, track.peak_cat, track.zcr, track.zcr_std, track.nrg, track.nrg_std, track.pow, track.pow_std, track.acr, track.acr_std, track.acr_lag, track.acr_lag_std, track.cent, track.cent_std, track.flx, track.flx_std, track.rlf, track.rlf_std, track.eoe, track.eoe_std, track.eoe_min, track.mfcc_0, track.mfcc_1, track.mfcc_2, track.mfcc_3, track.mfcc_4, track.mfcc_5, track.mfcc_6, track.mfcc_7, track.mfcc_8, track.mfcc_9, track.mfcc_10, track.mfcc_11, track.mfcc_12, track.chr_0, track.chr_1, track.chr_2, track.chr_3, track.chr_4, track.chr_5, track.chr_6, track.chr_7, track.chr_8, track.chr_9, track.chr_10, track.chr_11, track.mfcc_0_std, track.mfcc_1_std, track.mfcc_2_std, track.mfcc_3_std, track.mfcc_4_std, track.mfcc_5_std, track.mfcc_6_std, track.mfcc_7_std, track.mfcc_8_std, track.mfcc_9_std, track.mfcc_10_std, track.mfcc_11_std, track.mfcc_12_std, track.chr_0_std, track.chr_1_std, track.chr_2_std, track.chr_3_std, track.chr_4_std, track.chr_5_std, track.chr_6_std, track.chr_7_std, track.chr_8_std, track.chr_9_std, track.chr_10_std, track.chr_11_std , artist.is_male, artist.is_female, artist.is_group, artist.german, artist.american, artist.other_country, artist.total_years, artist.breaking_years, artist.life_span, artist.genre_electronic, artist.genre_pop, artist.genre_hiphop, artist.genre_rock, artist.genre_other, artist.followers, artist.listener, artist.play_count, artist.popularity, artist.mean_chart_peak_1, artist.mean_chart_peak_2, artist.mean_chart_peak_3, artist.mean_chart_peak_4, artist.mean_chart_peak_5, artist.mean_chart_peak_6, artist.mean_chart_weeks, artist.total_chart_weeks, artist.mean_album_chart_peak_0, artist.mean_album_chart_peak_1, artist.mean_album_chart_peak_2, artist.mean_album_chart_peak_3, artist.mean_album_chart_peak_4, artist.mean_album_chart_peak_5, artist.mean_album_chart_peak_6, artist.mean_album_chart_weeks, artist.total_album_chart_weeks FROM track JOIN artist ON track.artist_id = artist.id WHERE track.error = 0 AND peak_cat = 0  ORDER BY RANDOM() LIMIT {}".format(
                    n_rows), con)

    charts = pd.read_sql_query(
            "SELECT track.genre_electronic, track.genre_pop, track.genre_hiphop, track.genre_rock,track.genre_other, track.is_1980s, track.is_1990s, track.is_2000s, track.is_2010s, track.is_other_decade,  track.length, track.peak_cat, track.zcr, track.zcr_std, track.nrg, track.nrg_std, track.pow, track.pow_std, track.acr, track.acr_std, track.acr_lag, track.acr_lag_std, track.cent, track.cent_std, track.flx, track.flx_std, track.rlf, track.rlf_std, track.eoe, track.eoe_std, track.eoe_min, track.mfcc_0, track.mfcc_1, track.mfcc_2, track.mfcc_3, track.mfcc_4, track.mfcc_5, track.mfcc_6, track.mfcc_7, track.mfcc_8, track.mfcc_9, track.mfcc_10, track.mfcc_11, track.mfcc_12, track.chr_0, track.chr_1, track.chr_2, track.chr_3, track.chr_4, track.chr_5, track.chr_6, track.chr_7, track.chr_8, track.chr_9, track.chr_10, track.chr_11,track.mfcc_0_std, track.mfcc_1_std, track.mfcc_2_std, track.mfcc_3_std, track.mfcc_4_std, track.mfcc_5_std, track.mfcc_6_std, track.mfcc_7_std, track.mfcc_8_std, track.mfcc_9_std, track.mfcc_10_std, track.mfcc_11_std, track.mfcc_12_std, track.chr_0_std, track.chr_1_std, track.chr_2_std, track.chr_3_std, track.chr_4_std, track.chr_5_std, track.chr_6_std, track.chr_7_std, track.chr_8_std, track.chr_9_std, track.chr_10_std, track.chr_11_std , artist.is_male, artist.is_female, artist.is_group, artist.german, artist.american, artist.other_country, artist.total_years, artist.breaking_years, artist.life_span, artist.genre_electronic, artist.genre_pop, artist.genre_hiphop, artist.genre_rock, artist.genre_other, artist.followers, artist.listener, artist.play_count, artist.popularity, artist.mean_chart_peak, artist.mean_chart_weeks, artist.total_chart_weeks FROM track JOIN artist ON track.artist_id = artist.id WHERE track.error = 0 AND peak_cat != 0 ORDER BY RANDOM() LIMIT {}".format(
                    n_rows), con)

    complete_set = pd.concat([non_charts, charts])
    targets = complete_set['peak_cat']
    complete_set.drop('peak_cat', axis=1, inplace=True)

    return complete_set, targets


def selectData(id):
    data = pd.read_sql_query(
            "SELECT track.genre_electronic AS track_genre_electronic , track.genre_pop AS track_genre_pop, track.genre_hiphop AS track_genre_hiphop , track.genre_rock AS track_genre_rock, track.genre_other AS track_genre_other, track.is_1980s, track.is_1990s, track.is_2000s, track.is_2010s, track.is_other_decade,  track.length, track.peak_cat, track.zcr, track.zcr_std, track.nrg, track.nrg_std, track.pow, track.pow_std, track.acr, track.acr_std, track.acr_lag, track.acr_lag_std, track.cent, track.cent_std, track.flx, track.flx_std, track.rlf, track.rlf_std, track.eoe, track.eoe_std, track.eoe_min, track.mfcc_0, track.mfcc_1, track.mfcc_2, track.mfcc_3, track.mfcc_4, track.mfcc_5, track.mfcc_6, track.mfcc_7, track.mfcc_8, track.mfcc_9, track.mfcc_10, track.mfcc_11, track.mfcc_12, track.chr_0, track.chr_1, track.chr_2, track.chr_3, track.chr_4, track.chr_5, track.chr_6, track.chr_7, track.chr_8, track.chr_9, track.chr_10, track.chr_11, track.mfcc_0_std, track.mfcc_1_std, track.mfcc_2_std, track.mfcc_3_std, track.mfcc_4_std, track.mfcc_5_std, track.mfcc_6_std, track.mfcc_7_std, track.mfcc_8_std, track.mfcc_9_std, track.mfcc_10_std, track.mfcc_11_std, track.mfcc_12_std, track.chr_0_std, track.chr_1_std, track.chr_2_std, track.chr_3_std, track.chr_4_std, track.chr_5_std, track.chr_6_std, track.chr_7_std, track.chr_8_std, track.chr_9_std, track.chr_10_std, track.chr_11_std, artist.is_male, artist.is_female, artist.is_group, artist.german, artist.american, artist.other_country, artist.total_years, artist.life_span, artist.genre_electronic AS artist_genre_electronic, artist.genre_pop AS artist_genre_pop, artist.genre_hiphop AS artist_genre_hiphop, artist.genre_rock AS artist_genre_rock, artist.genre_other AS artist_genre_other, artist.followers, artist.listener, artist.play_count, artist.popularity, artist.mean_chart_peak_0, artist.mean_chart_peak_1, artist.mean_chart_peak_2, artist.mean_chart_peak_3, artist.mean_chart_peak_4, artist.mean_chart_peak_5, artist.mean_chart_peak_6, artist.mean_chart_weeks, artist.total_chart_weeks, artist.mean_album_chart_peak_0, artist.mean_album_chart_peak_1, artist.mean_album_chart_peak_2, artist.mean_album_chart_peak_3, artist.mean_album_chart_peak_4, artist.mean_album_chart_peak_5, artist.mean_album_chart_peak_6, artist.mean_album_chart_weeks, artist.total_album_chart_weeks FROM track JOIN artist ON track.artist_id = artist.id WHERE track.id = {}".format(
                    id),
            con)
    return data


def getData(n_rows, type=None, split=True, balanced=False):
    if n_rows is None:
        n_rows = -1
    if type == "md":
        query = "SELECT track.genre_electronic AS track_genre_electronic , track.genre_pop AS track_genre_pop, track.genre_hiphop AS track_genre_hiphop , track.genre_rock AS track_genre_rock, track.genre_other AS track_genre_other, track.is_1980s, track.is_1990s, track.is_2000s, track.is_2010s, track.is_other_decade,  track.length, track.peak_cat, artist.is_male, artist.is_female, artist.is_group, artist.german, artist.american, artist.other_country, artist.total_years, artist.life_span, artist.genre_electronic AS artist_genre_electronic, artist.genre_pop AS artist_genre_pop, artist.genre_hiphop AS artist_genre_hiphop, artist.genre_rock AS artist_genre_rock, artist.genre_other AS artist_genre_other, artist.followers, artist.listener, artist.play_count, artist.popularity, artist.mean_chart_peak_0, artist.mean_chart_peak_1, artist.mean_chart_peak_2, artist.mean_chart_peak_3, artist.mean_chart_peak_4, artist.mean_chart_peak_5, artist.mean_chart_peak_6, artist.mean_chart_weeks, artist.total_chart_weeks, artist.mean_album_chart_peak_0, artist.mean_album_chart_peak_1, artist.mean_album_chart_peak_2, artist.mean_album_chart_peak_3, artist.mean_album_chart_peak_4, artist.mean_album_chart_peak_5, artist.mean_album_chart_peak_6, artist.mean_album_chart_weeks, artist.total_album_chart_weeks FROM track JOIN artist ON track.artist_id = artist.id WHERE track.error = 0 AND artist.error=0  {} ORDER BY RANDOM() LIMIT {}"
    elif type == "mir":
        query = "SELECT track.zcr, track.zcr_std, track.nrg, track.nrg_std, track.pow, track.pow_std, track.acr, track.acr_std, track.acr_lag, track.acr_lag_std, track.cent, track.cent_std, track.flx, track.flx_std, track.rlf, track.rlf_std, track.eoe, track.eoe_std, track.eoe_min, track.mfcc_0, track.mfcc_1, track.mfcc_2, track.mfcc_3, track.mfcc_4, track.mfcc_5, track.mfcc_6, track.mfcc_7, track.mfcc_8, track.mfcc_9, track.mfcc_10, track.mfcc_11, track.mfcc_12, track.chr_0, track.chr_1, track.chr_2, track.chr_3, track.chr_4, track.chr_5, track.chr_6, track.chr_7, track.chr_8, track.chr_9, track.chr_10, track.chr_11, track.mfcc_0_std, track.mfcc_1_std, track.mfcc_2_std, track.mfcc_3_std, track.mfcc_4_std, track.mfcc_5_std, track.mfcc_6_std, track.mfcc_7_std, track.mfcc_8_std, track.mfcc_9_std, track.mfcc_10_std, track.mfcc_11_std, track.mfcc_12_std, track.chr_0_std, track.chr_1_std, track.chr_2_std, track.chr_3_std, track.chr_4_std, track.chr_5_std, track.chr_6_std, track.chr_7_std, track.chr_8_std, track.chr_9_std, track.chr_10_std, track.chr_11_std,track.peak_cat FROM track WHERE track.eoe NOT  NULL AND peak_cat NOT NULL {} ORDER BY RANDOM() LIMIT {}"
    else:
        query = "SELECT track.genre_electronic AS track_genre_electronic , track.genre_pop AS track_genre_pop, track.genre_hiphop AS track_genre_hiphop , track.genre_rock AS track_genre_rock, track.genre_other AS track_genre_other, track.is_1980s, track.is_1990s, track.is_2000s, track.is_2010s, track.is_other_decade,  track.length, track.peak_cat, track.zcr, track.zcr_std, track.nrg, track.nrg_std, track.pow, track.pow_std, track.acr, track.acr_std, track.acr_lag, track.acr_lag_std, track.cent, track.cent_std, track.flx, track.flx_std, track.rlf, track.rlf_std, track.eoe, track.eoe_std, track.eoe_min, track.mfcc_0, track.mfcc_1, track.mfcc_2, track.mfcc_3, track.mfcc_4, track.mfcc_5, track.mfcc_6, track.mfcc_7, track.mfcc_8, track.mfcc_9, track.mfcc_10, track.mfcc_11, track.mfcc_12, track.chr_0, track.chr_1, track.chr_2, track.chr_3, track.chr_4, track.chr_5, track.chr_6, track.chr_7, track.chr_8, track.chr_9, track.chr_10, track.chr_11, track.mfcc_0_std, track.mfcc_1_std, track.mfcc_2_std, track.mfcc_3_std, track.mfcc_4_std, track.mfcc_5_std, track.mfcc_6_std, track.mfcc_7_std, track.mfcc_8_std, track.mfcc_9_std, track.mfcc_10_std, track.mfcc_11_std, track.mfcc_12_std, track.chr_0_std, track.chr_1_std, track.chr_2_std, track.chr_3_std, track.chr_4_std, track.chr_5_std, track.chr_6_std, track.chr_7_std, track.chr_8_std, track.chr_9_std, track.chr_10_std, track.chr_11_std, artist.is_male, artist.is_female, artist.is_group, artist.german, artist.american, artist.other_country, artist.total_years, artist.life_span, artist.genre_electronic AS artist_genre_electronic, artist.genre_pop AS artist_genre_pop, artist.genre_hiphop AS artist_genre_hiphop, artist.genre_rock AS artist_genre_rock, artist.genre_other AS artist_genre_other, artist.followers, artist.listener, artist.play_count, artist.popularity, artist.mean_chart_peak_0, artist.mean_chart_peak_1, artist.mean_chart_peak_2, artist.mean_chart_peak_3, artist.mean_chart_peak_4, artist.mean_chart_peak_5, artist.mean_chart_peak_6, artist.mean_chart_weeks, artist.total_chart_weeks, artist.mean_album_chart_peak_0, artist.mean_album_chart_peak_1, artist.mean_album_chart_peak_2, artist.mean_album_chart_peak_3, artist.mean_album_chart_peak_4, artist.mean_album_chart_peak_5, artist.mean_album_chart_peak_6, artist.mean_album_chart_weeks, artist.total_album_chart_weeks FROM track JOIN artist ON track.artist_id = artist.id WHERE track.error = 0 AND artist.error=0 AND track.zcr IS NOT NULL {} ORDER BY RANDOM() LIMIT {}"

    if balanced:
        n_rows = int(n_rows / 7)
        # data = pd.DataFrame(np.zeros(shape=(116,n_rows)))
        data = pd.read_sql_query(
                query.format("AND peak_cat= {}".format(0), n_rows),
                con)
        data = data.append(pd.read_sql_query(
                query.format("AND peak_cat= {}".format(1), n_rows),
                con))
        data = data.append(pd.read_sql_query(
                query.format("AND peak_cat= {}".format(2), n_rows),
                con))
        data = data.append(pd.read_sql_query(
                query.format("AND peak_cat= {}".format(3), n_rows),
                con))
        data = data.append(pd.read_sql_query(
                query.format("AND peak_cat= {}".format(4), n_rows),
                con))
        data = data.append(pd.read_sql_query(
                query.format("AND peak_cat= {}".format(5), n_rows),
                con))
        data = data.append(pd.read_sql_query(
                query.format("AND peak_cat= {}".format(6), n_rows),
                con))
    else:
        data = pd.read_sql_query(
                query.format("", n_rows),
                con)

    if not split:
        return data
    targets = data['peak_cat']
    data.drop('peak_cat', axis=1, inplace=True)
    return data, targets


def impute(data):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return imp.fit_transform(data)


def plot(clf, feature_names, class_names, file):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names, class_names=class_names, filled=True,
                         rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(file)


def plot_lines(data, labels, xlabel, ylabel, title, suptitle, conf=None, additionals=None, path=None, ls='-'):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    for i, d in enumerate(data):
        ax.plot(range(1, len(d) + 1), d, color=colors[i], ls=ls, label=labels[i])
    legend = ax.legend(framealpha=.3)

    if conf is not None:
        import matplotlib.offsetbox as offsetbox
        keys = []
        vals = []
        for k, v in conf.items():
            if k != 'features':
                if v is None:
                    v = "none"
                elif type(v) == bool:
                    if v == True:
                        v = "true"
                    else:
                        v = "false"
                if k == 'n_input':
                    keys.append(offsetbox.TextArea(r"$features$"))
                    vals.append(offsetbox.TextArea(r"${}$".format(v), textprops={'size': 10}))
                elif k == 'units':
                    keys.append(offsetbox.TextArea(r"$hidden\_layers$"))
                    vals.append(offsetbox.TextArea(r"${}$".format(len(v)), textprops={'size': 10}))
                    for l, layer in enumerate(v):
                        keys.append(offsetbox.TextArea(r"$units_{HL{%d}}$" % (l)))
                        vals.append(offsetbox.TextArea(r"${}$".format(layer), textprops={'size': 10}))
                elif k == 'learning_rate':
                    keys.append(offsetbox.TextArea(r"$\eta$"))
                    vals.append(offsetbox.TextArea(r"${}$".format(v), textprops={'size': 10}))
                else:
                    keys.append(offsetbox.TextArea(r"${}$".format(k.replace("_", "\_"))))
                    vals.append(offsetbox.TextArea(r"${}$".format(v), textprops={'size': 10}))

        if additionals is not None:
            keys.append(offsetbox.TextArea(r"$E_{min, train}$", textprops={'color': colors[0]}))
            keys.append(offsetbox.TextArea(r"$E_{min, valid}$", textprops={'color': colors[1]}))
            vals.append(offsetbox.TextArea(r"${:.2e} {} (epoch {})$".format(additionals[0][1], " ", additionals[0][0]),
                                           textprops={'color': colors[0], 'size': 10, 'weight': 'bold'}))
            vals.append(offsetbox.TextArea(r"${:.2e} {} (epoch {})$".format(additionals[1][1], " ", additionals[1][0]),
                                           textprops={'color': colors[1], 'size': 10, 'weight': 'bold'}))

        vp1 = offsetbox.VPacker(children=keys, align="left", pad=0, sep=3)
        vp2 = offsetbox.VPacker(children=vals, align="right", pad=0, sep=5)
        hp = offsetbox.HPacker(children=(vp1, vp2), align="right", pad=5.76, sep=28.8)
        box = legend._legend_box
        box.get_children()[1].get_children()[0].get_children().append(hp)
        box.set_figure(box.figure)

    if additionals is not None:
        add = np.array(additionals).T
        plt.vlines(x=add[0], ymin=0, ymax=add[1], colors=colors, linestyles='dotted')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel not in ylabels else ylabels[ylabel])
    plt.xlim(xmax=len(data[0]))
    plt.title(title)
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.show()
