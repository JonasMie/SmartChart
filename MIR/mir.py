from __future__ import division

import pandas as pd
from marsyas_util import *

FEATURES_FILE = os.path.join('features', 'mir.csv')


def marsyas_analyse(input_filename, winSize=512, n_mfcc=13, n_chroma=12):
    print "Analyse File"

    csv_results = os.path.join('test', 'main', 'csv', 'main.csv')
    time_domain = [
        "Fanout/timeDomainSeries",
        [
            "ZeroCrossings/zcr",
            "Energy/eng",
            "Power/pow",
            ["Series/acrMax",
             [
                 "AutoCorrelation/acr",
                 "AbsMax/max"  # TODO: max? min? mean?
             ]
             ],
            [
                "Series/amdfMean",
                [
                    "AMDF/amdf",  # TODO: Windowing?
                    "Mean/amdf_mean"
                ]
            ]
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
            "Gain/gain",
            [
                "Fanout/domains",
                [
                    time_domain,
                    frequency_domain,
                ]
            ],
            "CsvSink/dest"
        ]
    ]
    net = create(spec)
    snet = mar_refs(spec)
    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)

    dest = net.getControl("CsvSink/dest/mrs_string/filename")
    dest.setValue_string(csv_results)
    inSamples = net.getControl("mrs_natural/inSamples")
    inSamples.setValue_natural(winSize)
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()
    nSamples = net.getControl(snet["src"] + "/mrs_natural/size").to_natural()

    '''
    Check if mono, stereo or even multi-channeled
    '''
    channels = net.getControl("SoundFileSource/src/mrs_natural/onObservations").to_natural()
    if (channels is not 2):
        raise NameError('Sorry, only Stereo-Recordings supported.')
    time_features = list()
    nTimeFeatures = 5
    for i in range(channels):
        time_features.insert(i, "zcr_{}".format(i))
        time_features.insert(i + channels, "nrg_{}".format(i))
        time_features.insert(i + channels * 2, "pow_{}".format(i))
        time_features.insert(i + channels * 3, "acr_{}".format(i))
        time_features.insert(i + channels * 4, "amdf_{}".format(i))

    mfcc_features = []
    chroma_features = []
    for i in range(max(n_mfcc, n_chroma, channels)):
        if i < n_mfcc:
            mfcc_features.append("mffc_{}".format(i))
        if i < n_chroma:
            chroma_features.append("chroma_{}".format(i))

    features = time_features + ["cent_0", "flx_0", "rlf_0"] + mfcc_features + chroma_features

    nFeatures = nTimeFeatures * channels + 3 + 13 + 12  # 13 Mel-Frequency Cepstral Coefficients & 12-sized Chroma Vector

    results = pd.DataFrame(np.zeros(shape=(math.ceil(nSamples / winSize), nFeatures)), columns=features)

    '''
    Gain hat Defaultwert von 1.0, wird hier verwendet, da sonst alle Werte 0.0 sind. Warum ??
    Direkt das Eingangssignal: Nur jeweils die Daten des aktuellen Windows ( Laenge = winSize),
    daher liefert control2array(net,"SoundFileSource/src/mrs_realvec/processedData") einen 512-Array

    CSV-Sink: Daten aller Frames werden konkateniert und am Ende gesammelt in CSV geschrieben,
    daher liefert die Serie mit CsvSink einen 5632-Array (winSize * frame_num+1)
    '''

    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")
    i = 0

    while notempty.to_bool():
        net.tick()
        results.iloc[i] = realvec2array(net.getControl("mrs_realvec/processedData").to_realvec())[0]
        i += 1

    return pd.DataFrame(results.mean())


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


def entropy_of_energy(input_filename, winSize=512, nSubFrames=8):
    """
    Since marsyas does not provide anything like power-entropy, I had to write the function
    :param input_filename:
    :param winSize:
    :param nSubFrames:
    """
    print "ZeroCrossings"
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
        if fft_size is not subWindowsLen * nSubWindows:
            window = window[:int(subWindowsLen * nSubWindows)]
        subWindows = window.reshape(nSubWindows, subWindowsLen)
        subEnergyPerc = [(subWindow ** 2).sum() / totalE for subWindow in subWindows]
        spectral_entropies[i] = -(subEnergyPerc * log2(subEnergyPerc)).sum()  # TODO: what if subEnergy == 0?

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


def run():
    file = os.path.join('test.mp3')
    # add_features = additional_features(file)
    features = marsyas_analyse(file).T
    features.to_csv(FEATURES_FILE, mode='a', index=False, header=None)