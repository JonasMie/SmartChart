from __future__ import division

import pandas as pd
from marsyas_util import *
from termcolor import colored

import utils

FEATURES_FILE = os.path.join('features', 'mir.csv')

error_obj = {
    'acr': None, 'acr_std': None,
    'acr_lag': None, 'acr_lag_std': None,
    'amdf': None, 'amdf_std': None,
    'zcr': None, 'zcr_std': None,
    'nrg': None, 'nrg_std': None,
    'pow': None, 'pow_std': None,
    'cent': None, 'cent_std': None,
    'flx': None, 'flx_std': None,
    'rlf': None, 'rlf_std': None,
    'mfcc_0': None, 'mfcc_0_std': None,
    'mfcc_1': None, 'mfcc_1_std': None,
    'mfcc_2': None, 'mfcc_2_std': None,
    'mfcc_3': None, 'mfcc_3_std': None,
    'mfcc_4': None, 'mfcc_4_std': None,
    'mfcc_5': None, 'mfcc_5_std': None,
    'mfcc_6': None, 'mfcc_6_std': None,
    'mfcc_7': None, 'mfcc_7_std': None,
    'mfcc_8': None, 'mfcc_8_std': None,
    'mfcc_9': None, 'mfcc_9_std': None,
    'mfcc_10': None, 'mfcc_10_std': None,
    'mfcc_11': None, 'mfcc_11_std': None,
    'mfcc_12': None, 'mfcc_12_std': None,
    'chr_0': None, 'chr_0_std': None,
    'chr_1': None, 'chr_1_std': None,
    'chr_2': None, 'chr_2_std': None,
    'chr_3': None, 'chr_3_std': None,
    'chr_4': None, 'chr_4_std': None,
    'chr_5': None, 'chr_5_std': None,
    'chr_6': None, 'chr_6_std': None,
    'chr_7': None, 'chr_7_std': None,
    'chr_8': None, 'chr_8_std': None,
    'chr_9': None, 'chr_9_std': None,
    'chr_10': None, 'chr_10_std': None,
    'chr_11': None, 'chr_11_std': None,
    'eoe': None, 'eoe_std': None, 'eoe_min': None
}

def entropy_of_energy(signal, winSize=512, nSubFrames=8):
    subWinSize = int(winSize / nSubFrames)
    entropies = zeros(len(signal))
    i = 0
    for channel in signal:
        if (np.count_nonzero(channel)) == 0:
            entropies[i] = 3.
        else:
            subEnergies = zeros(nSubFrames)
            energy = (channel ** 2).sum()
            swf = channel.reshape(-1, subWinSize)

            j = 0
            for subFrame in swf:
                subEnergies[j] = ((subFrame ** 2).sum()) / (energy + np.finfo(float).eps)
                j += 1
            entropies[i] = (-np.nansum(subEnergies * np.log2(subEnergies + np.finfo(float).eps)))
        i += 1
    return entropies.min()


def marsyas_analyse(input_filename, winSize=512, n_mfcc=13, n_chroma=12):
    utils.startProgress(u"| Analyzing file {}".format(input_filename))

    time_domain = [
        "Fanout/timeDomainSeries",
        [
            [
                "Series/acr_max",
                [
                    "AutoCorrelation/acr",
                    "Peaker/pkr",
                    "MaxArgMax/acr_max",
                    "Transposer/trans",
                    "Unfold/unfold"
                    # "Selector/sel",
                    # "Transposer/trans_1"
                ]
            ],
            [
                "Series/amdfMean",
                [
                    "AMDF/amdf",  # TODO: Windowing?
                    "Mean/amdf_mean"
                ]
            ],
            "ZeroCrossings/zcr",
            "Energy/eng",
            "Power/pow",

        ]
    ]
    frequency_domain = [
        "Series/frequencyDomain",
        [
            "Windowing/win",
            "Spectrum/spk",
            "PowerSpectrum/pspk",
            [
                "Fanout/afterFFT",
                [
                    "Centroid/ctr",
                    "Flux/flux",
                    "Rolloff/rlf",
                    "MFCC/mfcc",
                    "Chroma/chr",
                ]
            ],
        ],
    ]
    spec = [
        "Series/main",
        [
            "SoundFileSource/src",
            [
                "Fanout/fout",
                [
                    [
                        "Series/signal",
                        [
                            "Gain/gain",
                            "Transposer/transpo",
                        ]
                    ],
                    [
                        "Fanout/domains",
                        [
                            time_domain,
                            frequency_domain,
                        ]
                    ],
                ]
            ]
        ]
    ]

    net = create(spec)
    snet = mar_refs(spec)
    fname = net.getControl(snet['src'] + "/mrs_string/filename")
    fname.setValue_string(input_filename.encode('ascii', 'ignore'))

    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()
    nWindows = int(nSamples / winSize)

    if nSamples<=0:
        print colored("Negative sample amount, skipping file...", 'red')
        return error_obj
    # selector = net.getControl(snet["sel"] + "/mrs_natural/disable")
    # selector.setValue_natural(0)

    '''
    Check if mono, stereo or even multi-channeled
    '''
    channels = net.getControl(snet['src'] + "/mrs_natural/onObservations").to_natural()
    if channels != 1 and channels != 2:
        raise NameError('Sorry, only Mono & Stereo-Recordings supported.')

    time_features = list()
    nTimeFeatures = 6
    for i in range(channels):
        time_features.insert(i, "zcr_{}".format(i))
        time_features.insert(i + channels, "nrg_{}".format(i))
        time_features.insert(i + channels * 2, "pow_{}".format(i))
        time_features.insert(i + channels * 3, "acr_max{}".format(i))
        time_features.insert(i + channels * 4, "acr_argmax{}".format(i))
        time_features.insert(i + channels * 5, "amdf_{}".format(i))

    mfcc_features = []
    chroma_features = []
    for i in range(max(n_mfcc, n_chroma, channels)):
        if i < n_mfcc:
            mfcc_features.append("mffc_{}".format(i))
        if i < n_chroma:
            chroma_features.append("chroma_{}".format(i))

    features = time_features + ["cent_0", "flx_0", "rlf_0"] + mfcc_features + chroma_features + ["eoe"]
    nFeatures = nTimeFeatures * channels + 3 + 13 + 12 + 1  # 13 Mel-Frequency Cepstral Coefficients & 12-sized Chroma Vector & 1 EOE-Value
    results = pd.DataFrame(np.zeros(shape=(int(math.ceil(nSamples / winSize)), nFeatures)), columns=features)

    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")
    i = 0

    while notempty.to_bool():
        net.tick()
        res = realvec2array(net.getControl("mrs_realvec/processedData").to_realvec())
        try:
            results.iloc[i, :nFeatures - 1] = res[0, winSize:]
        except IndexError:
            print colored("Index error, skipping file...", 'red')
            results = None
            break
        results.iloc[i]["eoe"] = entropy_of_energy(res[:, :winSize])
        i += 1
        utils.progress(i / nWindows * 100)
    utils.endProgress()

    if results is None:
        return error_obj

    mean = results.mean()
    std = results.std()
    if channels == 1:
        res = {
            'acr': mean[0], 'acr_std': std[0],
            'acr_lag': mean[1], 'acr_lag_std': std[1],
            'amdf': mean[2], 'amdf_std': std[2],
            'zcr': mean[3], 'zcr_std': std[3],
            'nrg': mean[4], 'nrg_std': std[4],
            'pow': mean[5], 'pow_std': std[5],
            'cent': mean[6], 'cent_std': std[6],
            'flx': mean[7], 'flx_std': std[7],
            'rlf': mean[8], 'rlf_std': std[8],
            'mfcc_0': mean[9], 'mfcc_0_std': std[9],
            'mfcc_1': mean[10], 'mfcc_1_std': std[10],
            'mfcc_2': mean[11], 'mfcc_2_std': std[11],
            'mfcc_3': mean[12], 'mfcc_3_std': std[12],
            'mfcc_4': mean[13], 'mfcc_4_std': std[13],
            'mfcc_5': mean[14], 'mfcc_5_std': std[14],
            'mfcc_6': mean[15], 'mfcc_6_std': std[15],
            'mfcc_7': mean[16], 'mfcc_7_std': std[16],
            'mfcc_8': mean[17], 'mfcc_8_std': std[17],
            'mfcc_9': mean[18], 'mfcc_9_std': std[18],
            'mfcc_10': mean[19], 'mfcc_10_std': std[19],
            'mfcc_11': mean[20], 'mfcc_11_std': std[20],
            'mfcc_12': mean[21], 'mfcc_12_std': std[21],
            'chr_0': mean[22], 'chr_0_std': std[22],
            'chr_1': mean[23], 'chr_1_std': std[23],
            'chr_2': mean[24], 'chr_2_std': std[24],
            'chr_3': mean[25], 'chr_3_std': std[25],
            'chr_4': mean[26], 'chr_4_std': std[26],
            'chr_5': mean[27], 'chr_5_std': std[27],
            'chr_6': mean[28], 'chr_6_std': std[28],
            'chr_7': mean[29], 'chr_7_std': std[29],
            'chr_8': mean[30], 'chr_8_std': std[30],
            'chr_9': mean[31], 'chr_9_std': std[31],
            'chr_10': mean[32], 'chr_10_std': std[32],
            'chr_11': mean[33], 'chr_11_std': std[33],
            'eoe': mean[34], 'eoe_std': std[34], 'eoe_min': results['eoe'].min()
        }
    else:
        res = {
            'acr': np.mean((mean[0], mean[1])), 'acr_std': np.mean((std[0], std[1])),
            'acr_lag': np.mean((mean[2], mean[3])), 'acr_lag_std': np.mean((std[2], std[3])),
            'amdf': np.mean((mean[4], mean[5])), 'amdf_std': np.mean((std[4], std[5])),
            'zcr': np.mean((mean[6], mean[7])), 'zcr_std': np.mean((std[6], std[7])),
            'nrg': np.mean((mean[8], mean[9])), 'nrg_std': np.mean((std[8], std[9])),
            'pow': np.mean((mean[10], mean[11])), 'pow_std': np.mean((std[10], std[11])),
            'cent': mean[12], 'cent_std': std[12],
            'flx': mean[13], 'flx_std': std[13],
            'rlf': mean[14], 'rlf_std': std[14],
            'mfcc_0': mean[15], 'mfcc_0_std': std[15],
            'mfcc_1': mean[16], 'mfcc_1_std': std[16],
            'mfcc_2': mean[17], 'mfcc_2_std': std[17],
            'mfcc_3': mean[18], 'mfcc_3_std': std[18],
            'mfcc_4': mean[19], 'mfcc_4_std': std[19],
            'mfcc_5': mean[20], 'mfcc_5_std': std[20],
            'mfcc_6': mean[21], 'mfcc_6_std': std[21],
            'mfcc_7': mean[22], 'mfcc_7_std': std[22],
            'mfcc_8': mean[23], 'mfcc_8_std': std[23],
            'mfcc_9': mean[24], 'mfcc_9_std': std[24],
            'mfcc_10': mean[25], 'mfcc_10_std': std[25],
            'mfcc_11': mean[26], 'mfcc_11_std': std[26],
            'mfcc_12': mean[27], 'mfcc_12_std': std[27],
            'chr_0': mean[28], 'chr_0_std': std[28],
            'chr_1': mean[29], 'chr_1_std': std[29],
            'chr_2': mean[30], 'chr_2_std': std[30],
            'chr_3': mean[31], 'chr_3_std': std[31],
            'chr_4': mean[32], 'chr_4_std': std[32],
            'chr_5': mean[33], 'chr_5_std': std[33],
            'chr_6': mean[34], 'chr_6_std': std[34],
            'chr_7': mean[35], 'chr_7_std': std[35],
            'chr_8': mean[36], 'chr_8_std': std[36],
            'chr_9': mean[37], 'chr_9_std': std[37],
            'chr_10': mean[38], 'chr_10_std': std[38],
            'chr_11': mean[39], 'chr_11_std': std[39],
            'eoe': mean[40], 'eoe_std': std[40], 'eoe_min': results['eoe'].min()
        }

    return res


def zerocrossings(input_filename, frame_num=10, winSize=512):
    print "ZeroCrossings"
    spec = ["Series/zcExtract",
            ["SoundFileSource/src",
             "Gain/gain",
             ]
            ]
    net = create(spec)
    snet = mar_refs(spec)

    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)

    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()

    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")

    nElements = int(nSamples / winSize) + 1

    zcrs = zeros(nElements)
    zcrs_x = []
    zcrs_y = []
    num_zcrs = 0

    i = 0

    first_channel = np.zeros(shape=(nElements, 2))
    second_channel = np.zeros(shape=(nElements, 2))

    while notempty.to_bool():
        net.tick()
        waveform = control2array(net,
                                 "SoundFileSource/src/mrs_realvec/processedData").transpose()

        '''
        In the last frame, there may be less samples than the actual window size can contain
        '''

        first_channel[i, 0] = waveform[:, 0].min()
        first_channel[i, 1] = waveform[:, 0].max()
        # second_channel[i] = waveform[:,1].max
        # first_channel[i * winSize:i * winSize + length, 0] = waveform[:length, 0]
        # second_channel[i * winSize:i * winSize + length, 0] = waveform[:length, 1]

        for j in range(1, winSize):
            if (((waveform[j - 1, 0] > 0.0) and (waveform[j, 0] < 0.0)) or
                    ((waveform[j - 1, 0] < 0.0) and (waveform[j, 0] > 0.0))):
                zcrs_x.append((j + i * winSize) / winSize)
                zcrs_y.append(0.0)
                num_zcrs += 1
            if (((waveform[j - 1, 1] > 0.0) and (waveform[j, 1] < 0.0)) or
                    ((waveform[j - 1, 1] < 0.0) and (waveform[j, 1] > 0.0))):
                zcrs_x.append((j + i * winSize) / winSize)
                zcrs_y.append(0.0)
                num_zcrs += 1
        i += 1
    figure(1)

    # savetxt('C:\\Users\\Jonas\\Documents\\Studium\\Semester7\\BA\\test\\zcr\\csv\\y_max_values.csv', first_channel[:, 1], delimiter=",")

    title("Time Domain Zero Crossings " + "(" + str(num_zcrs) + ")")
    # plot the time domain waveform
    vlines(range(nElements), first_channel[:, 0], first_channel[:, 1])
    # plot(range(second_channel.shape[0]), second_channel[:, 0])
    # plot the zero-crossings

    # plot(zcrs_x, zcrs_y, 'rx', drawstyle='steps', markersize=8)
    # plot a line 0.0
    plot(zcrs)
    # label the axes
    xlabel("Time in Samples")
    ylabel("Amplitude")
    # save the figure
    output_filename = "C:\\Users\\Jonas\\Documents\\Studium\\Semester7\\BA\\test\\zcr\\img\\complete.png"
    savefig(output_filename)


def power(input_filename, winSize=512):
    print "Power"
    spec = ["Series/powerExtract",
            ["SoundFileSource/src",
             "Gain/gain",
             "Power/pow",
             "CsvSink/dest"
             ]
            ]
    net = create(spec)
    snet = mar_refs(spec)
    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)

    csv_path = os.path.join('test', 'power', 'csv', 'power.csv')
    img_path = os.path.join('test', 'power', 'img', 'rms_power.png')

    dest = net.getControl("CsvSink/dest/mrs_string/filename")
    dest.setValue_string(csv_path)
    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()

    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")

    while notempty.to_bool():
        net.tick()

    df = pd.read_csv(csv_path, sep=" ", names=["fc_power", "sc_power"])
    length = len(df["fc_power"])

    figure(1)
    title("RMS Power")

    plot(range(length), df["fc_power"])
    plot(range(length), df["fc_power"])

    # label the axes
    xlabel("Time in Samples")
    ylabel("RMS Power")
    # save the figure
    savefig(img_path)


def entropy_of_energy_test(input_filename, winSize=512, nSubFrames=8):
    """
    Since marsyas does not provide anything like power-entropy, I had to write the function
    :param input_filename:
    :param winSize:
    :param nSubFrames:
    """
    print "EoE"
    spec = ["Series/zcExtract",
            ["SoundFileSource/src",
             "Gain/gain",
             ]
            ]
    net = create(spec)
    snet = mar_refs(spec)

    subWinSize = winSize / nSubFrames
    csv_path = os.path.join('test', 'power_ent', 'csv', 'entropies.csv')

    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)

    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()
    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")

    entropoies = np.zeros(shape=(math.ceil(nSamples / winSize) + 1, 2))
    i = 0
    while notempty.to_bool():
        net.tick()

        first_subEnergies = zeros(nSubFrames)
        second_subEnergies = zeros(nSubFrames)
        waveform = control2array(net,
                                 "SoundFileSource/src/mrs_realvec/processedData")

        energy1 = (waveform[0] ** 2).sum()
        energy2 = (waveform[1] ** 2).sum()

        first_swf = waveform[0].reshape(-1, subWinSize)
        second_swf = waveform[1].reshape(-1, subWinSize)

        j = 0
        for subFrame in first_swf:
            first_subEnergies[j] = ((subFrame ** 2).sum()) / energy1
            j += 1
        j = 0
        for subFrame in second_swf:
            second_subEnergies[j] = ((subFrame ** 2).sum()) / energy2
            j += 1

        '''
        Since we may deal with really really small numbers ( up to e-09), the logarithm of this
        small number will result in -inf, which gives NAN when multiplied with the same number again.
        The casual sum()-method cannot deal with NANs, so the final result would be NAN, too.
        Fortunately numpy offers the nansum()-method, which simply ignores NANs.
        '''
        first_ent = np.nansum(first_subEnergies * np.log2(first_subEnergies))
        second_ent = np.nansum(second_subEnergies * np.log2(second_subEnergies))

        entropoies[i] = (-first_ent, -second_ent)
        i += 1
    pd.DataFrame(entropoies).to_csv(csv_path, index=False, header=None)


def spectrum(input_filename, frame_num=5, winSize=1024):
    spec = ["Series/pitchExtract",
            ["SoundFileSource/src",
             # "Windowing/win",
             "Spectrum/spk",
             # "InvSpectrum/ispk",
             # "PowerSpectrum/pspk",
             # "Gain/gain",
             "CsvSink/dest"
             ]
            ]
    net = create(spec)

    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)
    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)

    csv_results = os.path.join('test', 'spectrum', 'csv', 'first_window_spk_mono.csv')
    dest = net.getControl("CsvSink/dest/mrs_string/filename")
    dest.setValue_string(csv_results)

    # pspk = net.getControl("PowerSpectrum/pspk/mrs_string/spectrumType")
    # pspk.setValue_string("powerdensity")
    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")

    # while notempty.to_bool():
    net.tick()
    # data = control2array(net,
    #                      "SoundFileSource/src/mrs_realvec/processedData")
    # x = os.path.join('test', 'spectrum', 'csv', 'marsyas_data')
    # pd.DataFrame(data).to_csv(x, index=False, header=None)

    # if False:
    #     figure(1)
    #     data = net.getControl("PowerSpectrum/pspk/mrs_realvec/processedData").to_realvec()
    #     # restrict spectrum to first 93 bins corresponding approximately to 4000Hz
    #     spectrum = control2array(net, "PowerSpectrum/pspk/mrs_realvec/processedData", eo=93)
    #     # plot spectrum with frequency axis
    #     marplot(spectrum,
    #             x_label="Frequency in Hz",
    #             y_label="Power",
    #             plot_title="Power Spectrum",
    #             ex=4000)
    #     output_filename = os.path.splitext(input_filename)[0] + ".png"
    #     savefig(output_filename)


def spectrum_test(input_filename, winSize=512):
    spec = ["Series/pitchExtract",
            ["SoundFileSource/src",
             "Gain/gain",
             "CsvSink/dest"
             ]
            ]
    net = create(spec)
    snet = mar_refs(spec)

    csv_path = os.path.join('test', 'spectrum', 'csv', 'raw.csv')

    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)
    inSamples = net.getControl("mrs_natural/inSamples")
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()
    inSamples.setValue_natural(winSize)

    dest = net.getControl("CsvSink/dest/mrs_string/filename")
    dest.setValue_string(csv_path)

    # waveform = pd.DataFrame(control2array(net,
    #                                       "SoundFileSource/src/mrs_realvec/processedData"))
    #
    # waveform.to_csv(csv_path, index=False, header=None)
    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")

    while notempty.to_bool():
        net.tick()


def spectral_centroid(winSize=512):
    csv_path1 = os.path.join('test', 'spectrum', 'csv', 'first_window_mono.csv')
    df = pd.read_csv(csv_path1, sep=" ", names=["fc"])
    values = df["fc"].iloc[winSize:]

    fft = abs(np.fft.rfft(values))
    fft = fft / fft.max()

    fft_size = len(fft)

    freq = arange(0, winSize / 2 + 1)
    return ((fft * freq).sum() / fft.sum()) / fft_size


def spectral_spread(fft_df, winSize=512):
    fft_size = winSize / 2 + 1
    spectral_cent_and_spread = np.zeros(shape=(fft_df.shape[0], 2))

    for i, window in fft_df.iterrows():
        fft = window / window.max()

        sum = fft.sum()
        freq = arange(0, fft_size)
        if sum > 0:
            centroid = ((fft * freq).sum() / sum) / fft_size
            spread = (((((freq - centroid) ** 2) * fft).sum()) / sum) ** .5
        else:
            centroid = .5
            spread = 0  # TODO
        spectral_cent_and_spread[i, :] = (centroid, spread)

    return spectral_cent_and_spread


def spectral_entropy(fft_df, winSize=512, nSubWindows=8):
    fft_size = fft_df.shape[1]
    spectral_entropies = np.zeros(fft_df.shape[0])
    for i, window in fft_df.iterrows():
        totalE = (window ** 2).sum()
        subWindowsLen = math.floor(fft_size / nSubWindows)
        '''
        The fft_size is probably 2^n+1, but we need an even number of fft values, so its likely to ignore
         the last fft coefficient
         '''
        if fft_size != subWindowsLen * nSubWindows:
            window = window[:int(subWindowsLen * nSubWindows)]
        subWindows = window.reshape(nSubWindows, subWindowsLen)
        subEnergyPerc = [(subWindow ** 2).sum() / totalE for subWindow in subWindows]
        spectral_entropies[i] = -(subEnergyPerc * log2(subEnergyPerc)).sum()

    return spectral_entropies


def additional_features(input_filename, winSize=512):
    spec = ["Series/main",
            ["SoundFileSource/src",
             "Spectrum/spk",
             "PowerSpectrum/pspk"
             ]
            ]
    net = create(spec)
    snet = mar_refs(spec)
    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)

    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()

    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")
    fft_df = pd.DataFrame(np.zeros(shape=(math.ceil(nSamples / winSize), winSize / 2 + 1)))
    i = 0
    while notempty.to_bool():
        net.tick()
        fft_df.iloc[i] = control2array(net, "mrs_realvec/processedData").T
        i += 1

    return spectral_spread(fft_df, winSize), spectral_entropy(fft_df)


def flux():
    size = 512
    csv_path1 = os.path.join('test', 'spectrum', 'csv', 'first_window_mono.csv')
    df = pd.read_csv(csv_path1, sep=" ", names=["fc"])

    window0 = df["fc"].head(size)
    window1 = df["fc"].iloc[size:]

    fft0 = abs(np.fft.rfft(window0)) + 1e-20
    fft1 = abs(np.fft.rfft(window1)) + 1e-20

    fft0 = np.log((fft0 / fft0.sum()))
    fft1 = np.log((fft1 / fft1.sum()))

    x = ((fft1 - fft0) ** 2)
    max = x.max()

    print x.sum() / (max * 257)


def rolloff(winSize=512):
    cumulativeE = 0
    totalE = 0
    nSamples = 0
    counter = 0
    rolloff = .9

    csv_path1 = os.path.join('test', 'spectrum', 'csv', 'first_window_mono.csv')
    df = pd.read_csv(csv_path1, sep=" ", names=["fc"])
    window = df["fc"].head(winSize)

    fft = abs(np.fft.rfft(window))
    totalE = (fft ** 2).sum()

    while cumulativeE < totalE * rolloff:
        cumulativeE += fft[counter] ** 2
        counter += 1

    print ((counter - 1) / len(fft))


def test(input_filename, winSize=512):
    spec = ["Series/main",
            ["SoundFileSource/src",
             # ["Fanout/fno", [
             "AutoCorrelation/acr",
             # "Spectrum/spc"
             # ]],

             # "Peaker/pkr",
             # "MaxArgMax/max"
             ]
            ]
    net = create(spec)
    snet = mar_refs(spec)
    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)

    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()

    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")
    blag_m = np.zeros(nSamples / winSize + 1)
    HR_m = np.zeros(nSamples / winSize + 1)
    f0_m = np.zeros(nSamples / winSize + 1)
    i = 0
    blag_ = 0
    while notempty.to_bool():
        net.tick()
        res = realvec2array(net.getControl("mrs_realvec/processedData").to_realvec())
        window = res.T[0]
        blag = window[1]
        HR = window[0]
        f0 = 48000 / blag
        blag_ += blag
        HR_m[i] = HR
        f0_m[i] = f0
        print blag, HR, f0
        i += 1
    print "\n------\n"
    print blag_m.mean(), HR_m.mean(), f0_m.mean()
    print blag_, i
    print 48000 / (blag_ / i)


def acr(input_filename, winSize=512):
    spec = ["Series/main",
            ["SoundFileSource/src",
             ]
            ]
    net = create(spec)
    snet = mar_refs(spec)
    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)

    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()

    blag_m = np.zeros(nSamples / winSize + 1)
    HR_m = np.zeros(nSamples / winSize + 1)
    f0_m = np.zeros(nSamples / winSize + 1)
    i_ = 0
    x = 0
    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")
    while notempty.to_bool():
        net.tick()
        window = realvec2array(net.getControl("mrs_realvec/processedData").to_realvec()).T[0]
        Fs = 48000

        M = 0.016 * Fs
        R = np.correlate(window, window, mode='full')
        g = R[len(window) - 1]

        R = R[len(window):]
        i = 1

        m0 = len(R)
        while i < len(R):
            if R[i] < 0 and R[i - 1] >= 0:
                m0 = i
                break
            i += 1

        if M > len(R):
            M = len(R)

        Gamma = np.zeros(M)
        CSum = np.cumsum(window ** 2)

        Gamma[m0 - 1:M - 1] = R[m0 - 1:M - 1] / (g * CSum[-m0 - 1:-M - 1:-1]) ** .5

        blag = Gamma[1:].argmax() + 1
        HR = Gamma[blag]

        f0 = Fs / blag

        blag_m[i_] = blag
        HR_m[i_] = HR
        f0_m[i_] = f0
        x += blag
        i_ += 1
        print blag, HR, f0
    print "\n------\n"
    print blag_m.mean(), HR_m.mean(), f0_m.mean()
    print x, i_


def run():
    file = os.path.join(
            '/Users/jonas/Dropbox/BA/Training/009 Sound System/009 Sound System/Dreamscape(.mp3')
    # add_features = additional_features(file)
    # features = marsyas_analyse(file).T
    # features.to_csv(FEATURES_FILE, mode='a', index=False, header=None)
    test(file)
    # acr(file)
