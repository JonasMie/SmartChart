import pandas as pd
from marsyas_util import *

zcr_path = os.path.join('test', 'zcr', 'csv', 'zcr.csv')


def analyse(input_filename, winSize=512):
    print "Analyse File"

    csv_results = os.path.join('test', 'main', 'csv', 'main.csv')
    time_domain = ["Fanout/timeDomain", [
        "ZeroCrossings/zcr",
        "Energy/eng",
        "Power/pow"
    ]]
    spec = ["Series/pitchExtract",
            ["SoundFileSource/src",
             "Gain/gain",
             time_domain,
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

    '''
    Gain hat Defaultwert von 1.0, wird hier verwendet, da sonst alle Werte 0.0 sind. Warum ??
    Direkt das Eingangssignal: Nur jeweils die Daten des aktuellen Windows ( Laenge = winSize),
    daher liefert control2array(net,"SoundFileSource/src/mrs_realvec/processedData") einen 512-Array

    CSV-Sink: Daten aller Frames werden konkateniert und am Ende gesammelt in CSV geschrieben,
    daher liefert die Serie mit CsvSink einen 5632-Array (winSize * frame_num+1)
    '''

    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")

    while notempty.to_bool():
        net.tick()


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


def power(input_filename, winSize=51):
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


def spectrum(input_filename, frame_num=5, winSize=1024):  # 5 , 1024
    spec = ["Series/pitchExtract",
            ["SoundFileSource/src",
             "Windowing/win",
             "Spectrum/spk",
             "PowerSpectrum/pspk",
             "Gain/gain",
             "CsvSink/dest"
             ]
            ]
    net = create(spec)

    fname = net.getControl("SoundFileSource/src/mrs_string/filename")
    fname.setValue_string(input_filename)
    inSamples = net.getControl("mrs_natural/inSamples");
    inSamples.setValue_natural(winSize)

    csv_results = os.path.join('test', 'spectrum', 'csv', 'test.csv')
    dest = net.getControl("CsvSink/dest/mrs_string/filename")
    dest.setValue_string(csv_results)

    notempty = net.getControl("SoundFileSource/src/mrs_bool/hasData")

    while notempty.to_bool():
        net.tick()
        if False:
            figure(1);
            data = net.getControl("PowerSpectrum/pspk/mrs_realvec/processedData").to_realvec()
            # restrict spectrum to first 93 bins corresponding approximately to 4000Hz
            spectrum = control2array(net, "PowerSpectrum/pspk/mrs_realvec/processedData", eo=93);
            # plot spectrum with frequency axis
            marplot(spectrum,
                    x_label="Frequency in Hz",
                    y_label="Power",
                    plot_title="Power Spectrum",
                    ex=4000)
            output_filename = os.path.splitext(input_filename)[0] + ".png"
            savefig(output_filename)


def spectrum(input_filename, winSize=512):
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

    # Just to test
    #--------
    # size = 512 * 28019
    # csv_target = os.path.join('test', 'spectrum', 'csv', 'fft_res.csv')
    # csv_path = os.path.join('test', 'spectrum', 'csv', 'raw.csv')
    # df = pd.read_csv(csv_path, sep=" ", names=["fc", "sc"])
    # fft = pd.DataFrame(np.fft.fft(df["fc"]))
    # fft.to_csv(csv_target, index=False, header=None)
    # # (pd.DataFrame(np.fft.fft(df["sc"]))/size).to_csv(csv_target, index=False, header=None)
    # #--------
    # csv_path = os.path.join('test', 'spectrum', 'csv', 'fft_res.csv')
    # df = pd.read_csv(csv_path, names=["fc"])
    # print df.head(10)
    #--------


def run():
    file = os.path.join('test.wav')
    # analyse(file)
    # zerocrossings('C:\\Users\\Jonas\\Documents\\Studium\\Semester7\\BA\\test.mp3')
    # power(file)
    # entropy_of_energy(file)
    spectrum(file)
